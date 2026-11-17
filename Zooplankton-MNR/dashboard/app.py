"""Zooplankton dataset dashboard.

Turns the class-distribution exploration from summarize_data.ipynb into an
interactive Streamlit app. Counts are read straight from the image folders via
hierclassifier's ImageDataset, so this must run where that package and the data
directory are available.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from hierarchy import HIERARCHY_FILE, UNASSIGNED,  build_tree, load_hierarchy, unknown_leaves
 

# Set data
DATA_PATH = "/data/Hier-Zooplankton"
SUBDIRS = ["IssacData", "Quetico_training_data"]
MAX_CLASS_SIZE = 15000
IMAGE_RESOLUTION = 64
SEED = 6565

SELECTED_LIST = [
    "149um>", "Asplanchna", "Bosminidae", "Brachionidae", "Bubbles",'Bosmina_1',"Bythotrephes",
    "Calanoid", "Ceriodaphnia", "Chydoridae", "Chydorus", "Cladocera", "Conochilus",
    "Copepoda", "Cyclopoid", "Daphnia", "Debris", "Diaphonosoma", "Duplicates",
    "Eubosmina", "Exoskeleton", "Fiber_Hairlike", "Fiber_Squiggly", "Harpacticoid",
    "Holopedium", "Kellicottia", "Keratella", "Leptodora", "MultiClass",
    "Nauplius_Copepod", "Partial_Copepod", "Partial_Daphnia", "Partial_Leptodora",
    "Partial_Other", "Phytoplankton", "Plant_Matter", "Ploima", "Polyarthra",
    "Rotifer", "Sididae", "Synchaetidae", "Trichocerca", "Unknown", "Unknown_Small",
]

COLORS = {"Abundant": "#c2410c", "Rare": "#a16207", "Common": "#1d6f8c"}
PALETTE = [
    "#1d6f8c", "#7c3aed", "#c2410c", "#4d7c0f", "#be185d", "#0f766e",
    "#a16207", "#4338ca", "#b91c1c", "#0e7490", "#65a30d", "#9333ea",
]

# Hierarchy tab: one color per tree depth (root, level 1, level 2, ...).
LEVEL_COLORS = ["#6b7280", "#1d6f8c", "#7c3aed", "#c2410c", "#4d7c0f", "#be185d"]

st.set_page_config(page_title="Zooplankton Dataset", layout="wide")


# Data loading 
@st.cache_data(show_spinner="Scanning image folders...")
def load_data() -> pd.DataFrame:
    from hierclassifier.samples_setup import ImageDataset

    ds = ImageDataset(
        data_directory=DATA_PATH,
        data_subdirectories=SUBDIRS,
        class_names=SELECTED_LIST,
        max_class_size=MAX_CLASS_SIZE,
        image_resolution=IMAGE_RESOLUTION,
        image_transforms=None,
        format_file=".tif",
        seed=SEED,
    )
    return pd.DataFrame(ds.get_dataset_details())


# Sidebar 
if not Path(DATA_PATH).exists():
    st.error(f"Data directory not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data().copy()
except ImportError:
    st.error("Could not import `hierclassifier`. Install it in this environment first.")
    st.stop()

try:
    hier = load_hierarchy()
except (OSError, ValueError) as err:
    st.error(f"Could not read the class hierarchy: {err}")
    st.stop()

st.sidebar.header("Filters")
if st.sidebar.button("Rescan image folders"):
    load_data.clear()
    st.rerun()

rare_threshold = st.sidebar.number_input("Rare class: fewer than", 1, 5000, 100, step=10)
min_count = st.sidebar.slider("Minimum images per class", 0, int(df["Counts"].max()), 0)
selected = st.sidebar.multiselect("Classes (empty = all)", sorted(df["Class"]))
sort_by = st.sidebar.selectbox("Sort bars by", ["Count (high to low)", "Count (low to high)", "Name"])
log_scale = st.sidebar.checkbox("Log scale", value=False)
color_by = st.sidebar.radio("Color bars by", ["Status"], horizontal=True)

# Prepare 
total_all = int(df["Counts"].sum())
df["Share"] = df["Counts"] / total_all
df["Status"] = "Common"
df.loc[df["Counts"] < rare_threshold, "Status"] = "Rare"
df.loc[df["Counts"] >= MAX_CLASS_SIZE, "Status"] = "Abundant"

view = df[df["Counts"] >= min_count]
if selected:
    view = view[view["Class"].isin(selected)]

if sort_by == "Name":
    view = view.sort_values("Class")
else:
    view = view.sort_values("Counts", ascending=(sort_by == "Count (low to high)"))

# Header + KPIs 
st.title("Zooplankton image dataset")
st.caption(f"Source: {DATA_PATH} · classes capped at {MAX_CLASS_SIZE:,} images · seed {SEED}")

if view.empty:
    st.warning("No classes match the current filters.")
    st.stop()

nonzero = df[df["Counts"] > 0]
top = df.loc[df["Counts"].idxmax()]
c1, c2, c3 = st.columns(3)
c1.metric("Total images", f"{total_all:,}")
c2.metric("Classes", len(df))
c3.metric(f"Classes under {rare_threshold}", int((df["Counts"] < rare_threshold).sum()))

tab_dist, tab_imb, tab_hier, tab_data = st.tabs(["Distribution", "Imbalance", "Hierarchy", "Data"])
# --- Tab 1: distribution -----------------------------------------------------
with tab_dist:
    fig = px.bar(
        view, x="Class", y="Counts", color="Status",
        color_discrete_map=COLORS, text="Counts",
        hover_data={"Share": ":.2%", "Label": True} if "Label" in view else {"Share": ":.2%"},
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=520, xaxis_tickangle=-60, xaxis_title=None,
        yaxis_title="Images (log)" if log_scale else "Images",
        yaxis_type="log" if log_scale else "linear",
        legend_title=None, margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Orange bars hit the per-class cap, so their true size is probably larger. "
        "Amber bars fall under the rare-class threshold."
    )

# --- Tab 2: imbalance --------------------------------------------------------
with tab_imb:
    left, right = st.columns([3, 2])
    pareto = df.sort_values("Counts", ascending=False).reset_index(drop=True)
    pareto["Cumulative"] = pareto["Share"].cumsum()
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_bar(x=pareto["Class"], y=pareto["Counts"], name="Images", marker_color=COLORS["Common"])
    fig2.add_trace(
        go.Scatter(x=pareto["Class"], y=pareto["Cumulative"], name="Cumulative share",
                   mode="lines+markers", line=dict(color="#111827")),
        secondary_y=True,
    )
    fig2.update_yaxes(title_text="Images", secondary_y=False)
    fig2.update_yaxes(title_text="Cumulative share", tickformat=".0%", range=[0, 1.02], secondary_y=True)
    fig2.update_layout(height=480, xaxis_tickangle=-60, legend=dict(orientation="h", y=1.08))
    left.plotly_chart(fig2, use_container_width=True)

    k = int((pareto["Cumulative"] < 0.8).sum()) + 1
    right.markdown(f"**{k} of {len(df)} classes** make up 80% of all images.")
    right.markdown(f"**Rare classes (under {rare_threshold}):**")
    rare = df[df["Status"] == "Rare"].sort_values("Counts")[["Class", "Counts"]]
    right.dataframe(rare, hide_index=True, use_container_width=True, height=360)
    
# --- Tab 3: hierarchy -------------------------------------------------------
with tab_hier:
    all_counts = dict(zip(df["Class"], df["Counts"]))
    # Only classes that appear in the hierarchy; everything else is left out.
    counts = {c: n for c, n in all_counts.items() if c in hier.class_branch}
    tree = build_tree(hier, counts)
    typos = unknown_leaves(hier, all_counts)
    if typos:
        st.warning(f"In {HIERARCHY_FILE.name} but not in the dataset (typo?): " + ", ".join(typos))
    excluded = sorted(c for c in all_counts if c not in hier.class_branch)
    if excluded:
        with st.expander(f"{len(excluded)} dataset classes not in the hierarchy (excluded)"):
            st.write(", ".join(excluded))
    # Depth of each node in the tree (root = 0) -> one color per level.
    parent_of = dict(zip(tree["node"], tree["parent"]))

    def depth(node):
        d = 0
        while parent_of.get(node):
            node, d = parent_of[node], d + 1
        return d

    tree["depth"] = tree["node"].map(depth)
    level_name = {0: "Root"} | {d: f"Level {d}" for d in range(1, tree["depth"].max() + 1)}
    tree["level"] = tree["depth"].map(level_name)
    tree["color"] = tree["depth"].map(lambda d: LEVEL_COLORS[d % len(LEVEL_COLORS)])

    st.markdown(
        " &nbsp; ".join(
            f"<span style='display:inline-block;width:0.9em;height:0.9em;border-radius:2px;"
            f"background:{LEVEL_COLORS[d % len(LEVEL_COLORS)]};vertical-align:middle'></span> {name}"
            for d, name in level_name.items()
        ),
        unsafe_allow_html=True,
    )

    view_hier = st.radio("View", ["Sunburst", "Treemap", "Flowchart"], horizontal=True)
    if view_hier == "Flowchart":
        lines = [
            "digraph H {",
            "rankdir=LR; nodesep=0.15; ranksep=0.5;",
            'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor=white, penwidth=0];',
            'edge [color="#9ca3af", arrowsize=0.6];',
        ]
        for r in tree.itertuples():
            label = f"{r.node}\\n{r.total:,}"
            lines.append(f'"{r.node}" [label="{label}", fillcolor="{r.color}"];')
            if r.parent:
                lines.append(f'"{r.parent}" -> "{r.node}";')
        lines.append("}")
        st.graphviz_chart("\n".join(lines), use_container_width=True)
        st.caption("Each box shows the rolled-up image count (own images plus all descendants).")
    else:
        trace = go.Sunburst if view_hier == "Sunburst" else go.Treemap
        fig3 = go.Figure(trace(
            labels=tree["node"], parents=tree["parent"], values=tree["total"],
            branchvalues="total", marker=dict(colors=tree["color"]),
            customdata=tree[["own", "share", "level"]],
            hovertemplate="<b>%{label}</b> (%{customdata[2]})<br>Rolled-up: %{value:,} "
                          "(%{customdata[1]:.1%})<br>Own images: %{customdata[0]:,}<extra></extra>",
        ))
        fig3.update_layout(height=650, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Sizes are rolled-up totals (a node's own images plus all its descendants). "
            "For classes with children, such as Cladocera, the sector area beyond the "
            "children is the parent's own images."
        )
    st.caption(
        "Colors mark depth in the tree. A node that shares its parent's name is merged "
        "into it, so it takes the shallower level (e.g. Bubbles sits at Level 2). "
        "This tab ignores the sidebar filters."
    )

    groups = tree[tree["has_children"]].sort_values("total", ascending=False).copy()
    groups["share"] = groups["share"] * 100
    st.dataframe(
        groups.rename(columns={"node": "Group", "own": "Own images", "total": "Rolled-up", "share": "Share (%)"})
        [["Group", "Own images", "Rolled-up", "Share (%)"]],
        hide_index=True, use_container_width=True,
        column_config={"Share (%)": st.column_config.NumberColumn(format="%.1f")},
    )
    with st.expander(f"Edit {HIERARCHY_FILE.name} to change this tree"):
        st.code(HIERARCHY_FILE.read_text(), language="python")

# --- Tab 4: data -------------------------------------------------------------
with tab_data:
    shown = view[["Class", "Counts", "Share", "Status"]]
    st.dataframe(
        shown, hide_index=True, use_container_width=True,
        column_config={"Share": st.column_config.ProgressColumn(
            "Share", format="%.2f", min_value=0.0, max_value=float(df["Share"].max()))},
    )
    st.download_button("Download filtered table (CSV)", shown.to_csv(index=False), "class_counts_filtered.csv")