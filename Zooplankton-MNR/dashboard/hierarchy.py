"""Class hierarchy for the dashboard, built from hierarchy_config.py.

Edit hierarchy_config.py (coarse_names / groups, same layout as the training
code) to change the tree. The file is re-read on every dashboard rerun, so
edits show up without restarting Streamlit.

The per-level group lists become a parent/child tree:

    ROOT -> level-1 group -> level-2 group -> ... -> dataset class

Rules
- A node with the same name as its parent is merged into it. Level-2 "Bubbles"
  -> level-3 "Bubbles" -> class "Bubbles" becomes a single node, and a class
  that shares its group's name (Copepoda, Cladocera, Rotifer) keeps its own
  images and adds its children's on top.
- A class missing from a finer level hangs from the finest group it is in
  (e.g. Debris, listed only at level 1, sits directly under Zoop-No).
- Dataset classes missing from every level go under "Unassigned".
- The tree must be consistent: a class may appear in only one group per level,
  and every group must sit inside a single group of the level above.
  Violations raise a ValueError that names the offending class.

Run `python hierarchy.py` to print the resulting tree.
"""
import runpy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HIERARCHY_FILE = Path(__file__).with_name("hierarchy_config.py")
UNASSIGNED = "Unassigned"


@dataclass
class Hierarchy:
    root: str
    edges: list          # [(node, parent), ...] depth-first, config order
    class_branch: dict   # {class listed in the config: color branch}
    branches: list       # color branches in display order (without Unassigned)
    classes: list        # every class listed in the config


# --- Reading the config ------------------------------------------------------

def _read_config(path):
    try:
        cfg = runpy.run_path(str(path))
    except SyntaxError as err:
        raise ValueError(f"{path.name}, line {err.lineno}: {err.msg}") from err
    for key in ("coarse_names", "groups"):
        if key not in cfg:
            raise ValueError(f"{path.name} does not define `{key}`")
    return cfg


def _levels(cfg):
    """[(coarse_names, groups), ...] from coarsest (level 1) to finest."""
    names = list(cfg["coarse_names"])[::-1]
    groups = list(cfg["groups"])[::-1]
    if len(names) != len(groups):
        raise ValueError(
            f"coarse_names has {len(names)} levels but groups has {len(groups)}")
    if cfg.get("LEVELS", len(names)) != len(names):
        raise ValueError(f"LEVELS = {cfg['LEVELS']} but {len(names)} levels are listed")
    for lvl, (n, g) in enumerate(zip(names, groups), start=1):
        if len(n) != len(g):
            raise ValueError(
                f"Level {lvl}: {len(n)} coarse names but {len(g)} groups")
        dupes = sorted({x for x in n if list(n).count(x) > 1})
        if dupes:
            raise ValueError(f"Level {lvl}: duplicate group names {dupes}")
    return list(zip(names, groups))


def load_hierarchy(path=HIERARCHY_FILE) -> Hierarchy:
    cfg = _read_config(Path(path))
    root = cfg.get("ROOT", "Zooplankton")
    levels = _levels(cfg)
    color_level = cfg.get("COLOR_LEVEL", 1)
    if not 1 <= color_level <= len(levels):
        raise ValueError(f"COLOR_LEVEL must be between 1 and {len(levels)}")

    # Each class's group at every level (None where a level does not list it).
    paths = {}
    for lvl, (names, groups) in enumerate(levels):
        for gname, members in zip(names, groups):
            for c in members:
                path_ = paths.setdefault(c, [None] * len(levels))
                if path_[lvl] is not None:
                    raise ValueError(
                        f"Level {lvl + 1}: class {c!r} is in both "
                        f"{path_[lvl]!r} and {gname!r}")
                path_[lvl] = gname

    # Display order: level by level, groups in config order, then classes.
    rank = {root: 0}
    for names, groups in levels:
        for n in names:
            rank.setdefault(n, len(rank))
        for members in groups:
            for c in members:
                rank.setdefault(c, len(rank))

    # Parent links, merging nodes that share their parent's name.
    parent = {root: None}
    for cls, path_ in paths.items():
        prev = root
        for node in [g for g in path_ if g is not None] + [cls]:
            if node == prev:
                continue
            if parent.get(node, prev) != prev:
                raise ValueError(
                    f"{node!r} would have two parents, {parent[node]!r} and "
                    f"{prev!r} (while placing class {cls!r}). Check that its "
                    f"members sit in one group at the level above.")
            parent[node] = prev
            prev = node

    children = {}
    for node, par in parent.items():
        if par is not None:
            children.setdefault(par, []).append(node)
    edges = []

    def walk(node):
        edges.append((node, parent[node]))
        for ch in sorted(children.get(node, []), key=lambda x: rank[x]):
            walk(ch)

    walk(root)

    # Color branch of each class: its group at COLOR_LEVEL, or the finest
    # coarser group if that level does not list it.
    class_branch = {}
    for cls, path_ in paths.items():
        upto = [g for g in path_[:color_level] if g is not None]
        class_branch[cls] = upto[-1] if upto else UNASSIGNED
    color_names = set(levels[color_level - 1][0])
    branches = sorted(set(class_branch.values()) - {UNASSIGNED},
                      key=lambda x: (x not in color_names, rank[x]))

    return Hierarchy(root, edges, class_branch, branches, list(paths))


# --- Functions used by the dashboard ----------------------------------------

def _parent_map(hier, class_names=()):
    parent = dict(hier.edges)
    for c in class_names:
        parent.setdefault(c, UNASSIGNED)
    if UNASSIGNED in parent.values():
        parent[UNASSIGNED] = hier.root
    return parent


def branch_of(hier: Hierarchy, name: str) -> str:
    """Color branch of a class or node (its group at COLOR_LEVEL)."""
    if name in hier.class_branch:
        return hier.class_branch[name]
    parent = _parent_map(hier, [name])
    while name is not None and name not in hier.branches:
        name = parent.get(name)
    return name or UNASSIGNED


def unknown_leaves(hier: Hierarchy, counts: dict) -> list:
    """Classes listed in the config that the dataset does not have (typos?)."""
    return [c for c in hier.classes if c not in counts]


def build_tree(hier: Hierarchy, counts: dict) -> pd.DataFrame:
    """One row per node with its own count and its rolled-up total."""
    parent = _parent_map(hier, counts)
    children = {}
    for node, par in parent.items():
        if par is not None:
            children.setdefault(par, []).append(node)

    def total(node):
        return counts.get(node, 0) + sum(total(c) for c in children.get(node, []))

    rows = [
        {
            "node": node,
            "parent": par or "",
            "own": int(counts.get(node, 0)),
            "total": int(total(node)),
            "is_class": node in counts,
            "has_children": node in children,
        }
        for node, par in parent.items()
    ]
    tree = pd.DataFrame(rows)
    root_total = tree.loc[tree["node"] == hier.root, "total"].iloc[0]
    tree["share"] = tree["total"] / root_total if root_total else 0.0
    return tree


def print_tree(hier: Hierarchy = None):
    """Print the tree as indented text (handy after editing the config)."""
    hier = hier or load_hierarchy()
    parent = dict(hier.edges)
    for node, par in hier.edges:
        depth, p = 0, par
        while p is not None:
            depth, p = depth + 1, parent[p]
        tag = f"   [branch: {branch_of(hier, node)}]" if node in hier.class_branch else ""
        print("  " * depth + node + tag)


if __name__ == "__main__":
    print_tree()