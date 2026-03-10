import numpy as np
import random
import torch 

def set_seed(seed: int = 666):

    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_metrics(metrics_dict):

    """
    Returns new dictionary with the same keys and Tensor metric values converted to Python floats.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists 
            containing numbers or PyTorch tensors.

    Example:
        Input: {'loss': [tensor(0.5), tensor(0.3)], 'acc': [0.8, tensor(0.9)]}
        Output: {'loss': [0.5, 0.3], 'acc': [0.8, 0.9]}
    """

    cleaned = {}

    for key, values in metrics_dict.items():
        cleaned_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                cleaned_values.append(v.item())
            else:
                cleaned_values.append(float(v))
        cleaned[key] = cleaned_values

    return cleaned

def extract_metrics_hier(metrics_dict):

    """
     Returns new dictionary with the same keys and Tensor metric values converted to Python floats.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists 
            containing lists per level  of numbers or PyTorch tensors.

    Example:
        Input: {
            'loss': [[tensor(0.5), tensor(0.3)].[tensor(0.8), tensor(0.6)]], 
            'acc': [[0.8, tensor(0.9)],[0.5, tensor(0.1)]]
            }
        Output: {
            'loss': [[0.5, 0.3],[0.8,0.6]],
            'acc': [[0.8, 0.9],[0.5,0.1]]
            }

    """

    cleaned = {}

    for key, values in metrics_dict.items():
        size = len(values)
        if isinstance(values[0], list):
            cleaned_values = [[] for _ in range(size)]
            for l in range(size):   
                curr_val = values[l]
                for v in curr_val:
                    if isinstance(v, torch.Tensor):
                        cleaned_values[l].append(v.item())
                    else:
                        cleaned_values[l].append(float(v))
                        
        else: 
            cleaned_values = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    cleaned_values.append(v.item())
                else:
                    cleaned_values.append(float(v))
                                        
        cleaned[key] = cleaned_values

    return cleaned

def parent_children_dicts(hierarchy_names, hierarchy_groups):
    
    """
     Returns two lists with parent-child and child-parent dictionaries

    Args:
        hierarchy_names (list): A list of string lists, where each inner list defines the names of the groups at a given
                            coarser level. The order of the list is important: the fist corresponds to the finest grouping
                            and the last to the broadest. 
        hierarchy_groups (list): A list of string lists, where each inner list defines how the final nodes are grouped at a given
                       coarser level. The order of the list is important: the first corresponds to the finest
                       grouping, and the last to the broadest. 
        
    Output: 
        parent_maps : list[dict[str, str]]
            parent_maps[i]: child_label_at_level_(i+1) -> parent_label_at_level_i.
        children_groups : list[dict[str, list[str]]]
            children_groups[i][parent_label_at_level_i] -> list of child labels at level i+1.
   
    """
    
    levels = len(hierarchy_names)
        
    ALL_BOSMINA = ['Bosminidae', 'Bosmina_1', 'Eubosmina']
    ALL_FIBER = ['Fiber_Squiggly','Fiber_Hairlike']

    parents_maps = []
    children_groups = []

    for l in range(levels-1,0,-1):
        dict_p = {}
        dict_c = {}
        for parent, children in zip(hierarchy_names[l],hierarchy_groups[l]):
            
            valid_children_groups = hierarchy_names[l-1]
                    
            clean_children = []
            for c in children:
                if(c in ALL_BOSMINA):
                    c = 'Bosmina'
                elif(c in ALL_FIBER):
                    c = 'Fiber'
                clean_children.append(c)
                dict_p[c] = parent
            clean_children_filtered = [
                x for x in clean_children if x in valid_children_groups
            ]
            dict_c[parent] = sorted(set(clean_children_filtered))
            
        # save parents
        keys_to_keep = set(valid_children_groups)
        filtered_dict_p = {k: v for k, v in dict_p.items() if k in keys_to_keep}   
        parents_maps.append(filtered_dict_p)
        
        # save children 
        children_groups.append(dict_c)  
        
    return parents_maps, children_groups
    

def enforce_hierarchical_consistency(
    all_levels_pred, 
    confidences,
    parent_maps,
    children_groups
):
    """
    Enforce hierarchical consistency for an arbitrary number of levels.

    Parameters
    ----------
    all_levels_pred : list[str]
        Predicted labels, from coarsest (level 0) to finest (level L-1).
    confidences : list[dict[str, float]]
        Per-level confidence dictionaries, same order as all_levels_pred.
    parent_maps : list[dict[str, str]]
        parent_maps[i]: child_label_at_level_(i+1) -> parent_label_at_level_i.
    children_groups : list[dict[str, list[str]]]
        children_groups[i][parent_label_at_level_i] -> list of child labels at level i+1.

    Returns
    -------
    list[str]
        Corrected predictions with enforced parent–child consistency.
    """
    L = len(all_levels_pred)
    assert len(confidences) == L
    assert len(parent_maps) == L - 1
    assert len(children_groups) == L - 1

    # Work on a copy so we don't modify the input list in-place
    preds = list(all_levels_pred)

    # Go through each adjacent (parent, child) pair
    for parent_level in range(L - 1):
        child_level = parent_level + 1

        parent_pred = preds[parent_level]
        child_pred = preds[child_level]

        parent_conf = confidences[parent_level]
        child_conf = confidences[child_level]

        child_to_parent = parent_maps[parent_level]
        parent_to_children = children_groups[parent_level]

        # Only enforce if this child is in the mapping
        if child_pred in child_to_parent:
            expected_parent = child_to_parent[child_pred]

            if parent_pred != expected_parent:
                # Compare confidence of expected vs current parent
                conf_expected_parent = parent_conf.get(expected_parent, 0.0)
                conf_current_parent = parent_conf.get(parent_pred, 0.0)

                if conf_expected_parent >= conf_current_parent:
                    # Trust child → fix parent
                    preds[parent_level] = expected_parent
                else:
                    # Trust parent → adjust child to best child under that parent
                    possible_children = parent_to_children.get(parent_pred, [])
                    if possible_children:
                        best_child = max(
                            possible_children,
                            key=lambda c: child_conf.get(c, 0.0)
                        )
                        preds[child_level] = best_child

    return preds