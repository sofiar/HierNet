"""
Hyperparameter tuning for the hierarchical BCNN (zooplankton).

1. Tuning: runs BCNN_Model.gridsearch over HYPERPARAMETER_SEARCH_GRID and saves the best configuration
   to Models_results/tuning/TUNE_*.json. Set RUN_GRIDSEARCH = False to skip this and load the best
   hyperparameters from an existing TUNE_*.json instead.
2. Retraining with the best hyperparameters (weights / predictions / metadata saved like train_bcnn.py):
   - evaluation model ('_tuned'): same 70/10/20 split as tuning, used for the results figures
   - production model ('_production'): 90/05/05 split, the model to deploy
   Both run names are written back into the TUNE_*.json.

"""

import os
import json
import torch
from torchvision import transforms

from hierclassifier.extra_functions import set_seed
from hierclassifier.samples_setup import ImageDataset, HierImageDataset
from hierclassifier.bcnn_setup import BCNN_Model


# ################################################################################
#           ENVIRONMENT SET-UP
# ################################################################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
print(f'Using device: {device}')

# Specify paths
data_directory = '/data/Hier-Zooplankton'
data_subdirectories = ['IssacData','Quetico_training_data']
weights_directory = './pre_trained_weights'
results_directory = '/home/ruizsuar/Plankton-h-classifier/Models_results'

# Specify other environment variables
SEED = 666
set_seed(SEED)


# ################################################################################
#           LOAD DATA 
# ################################################################################

ZOOPLANKTON_CLASSES = [
    'Asplanchna', 'Debris', 'Bubbles', 'Exoskeleton', 'Fiber_Squiggly', 'Fiber_Hairlike',
    'Copepoda', 'Calanoid', 'Cladocera', 'Cyclopoid', 'Chydorus', 'Harpacticoid',
    'Holopedium', 'Kellicottia', 'Conochilus', 'Bosminidae', 'Bosmina_1', 'Eubosmina',
    'Daphnia', 'Rotifer', 'Nauplius_Copepod', 'Plant_Matter', 'Trichocerca'
]

MAX_CLASS_SIZE = 15000
RESOLUTION = 64

dataset = ImageDataset(
    data_directory = data_directory,
    data_subdirectories = data_subdirectories,
    class_names = ZOOPLANKTON_CLASSES,
    max_class_size = MAX_CLASS_SIZE,
    image_resolution = RESOLUTION,
    image_transforms = None,
    format_file = '.tif',
    seed = SEED
)

# Hierarchy (finest -> coarsest)
LEVELS = 3

coarse_names1 = ['Zoop-yes', 'Zoop-No']
groups1 = [
    ['Copepoda','Cladocera','Bosminidae', 'Bosmina_1','Eubosmina','Daphnia',
     'Cyclopoid','Harpacticoid','Holopedium','Calanoid','Rotifer',
     'Nauplius_Copepod','Trichocerca','Asplanchna', 'Kellicottia',
     'Conochilus', 'Chydorus'],
    ['Debris','Bubbles','Exoskeleton','Fiber_Squiggly',
     'Fiber_Hairlike','Plant_Matter']
]

coarse_names2 = ['Copepoda', 'Cladocera','Rotifer','Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter']
groups2 = [
    ['Copepoda','Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod'],
    ['Cladocera','Bosminidae','Bosmina_1','Eubosmina','Daphnia','Holopedium','Chydorus'],
    ['Rotifer','Trichocerca','Asplanchna','Conochilus','Kellicottia'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Squiggly','Fiber_Hairlike'],
    ['Plant_Matter']
]

coarse_names3 = [
    'Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod',
    'Bosmina','Holopedium','Daphnia','Chydorus',
    'Kellicottia','Conochilus','Asplanchna','Trichocerca',
    'Bubbles','Exoskeleton','Fiber','Plant_Matter'
]
groups3 = [
    ['Cyclopoid'], ['Calanoid'], ['Harpacticoid'], ['Nauplius_Copepod'],
    ['Bosminidae','Bosmina_1','Eubosmina'], ['Holopedium'], ['Daphnia'], ['Chydorus'],
    ['Kellicottia'], ['Conochilus'], ['Asplanchna'], ['Trichocerca'],
    ['Bubbles'], ['Exoskeleton'], ['Fiber_Squiggly','Fiber_Hairlike'], ['Plant_Matter']
]

coarse_names = [coarse_names3, coarse_names2, coarse_names1]
groups = [groups3, groups2, groups1]

hier_dataset = HierImageDataset(
    base_dataset = dataset,
    groups = groups,
    coarse_names = coarse_names
)
hier_dataset.print_dataset_details()

# Output size per level, coarsest first: [2, 7, 16]
dim_outputs = [len(names) for names in reversed(coarse_names)]


# ################################################################################
#           TRANSFORMS, SPLITS & DATA LOADERS
# ################################################################################

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
])

hier_dataset.append_image_transforms(image_transforms = train_transforms, replace = True)

TRAIN_PROP = 0.7
VAL_PROP = 0.1
TEST_PROP = 0.2

BATCH_SIZE = 64

train_split, val_split, test_split = hier_dataset.split_train_test_val(
    train_prop = TRAIN_PROP, val_prop = VAL_PROP, test_prop = TEST_PROP
)

train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size = BATCH_SIZE,
    train_indices = train_split,
    val_indices = val_split,
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None
)



# ################################################################################
#           HYPERPARAMETER TUNING (or load a previous result)
# ################################################################################

MODEL_NAME = 'densenet121'  # densenet121, resnet50, vgg16

tuning_directory = os.path.join(results_directory, 'tuning')
os.makedirs(tuning_directory, exist_ok = True)

RUN_GRIDSEARCH = False
TUNING_JSON = 'TUNE_BCNN_ZOOP_20261124_091236_densenet121_3.json'  

# 'alpha' weights the loss of each level (coarsest first) and must have LEVELS entries.
# With 'thresholds', alpha switches to the next list at each epoch threshold.
HYPERPARAMETER_SEARCH_GRID = {
    'loss_fn': [
        {'criterion': 'CrossEntropyLoss', 'alpha': [1/3, 1/3, 1/3], 'thresholds': None},
        {'criterion': 'CrossEntropyLoss', 'alpha': [0.2, 0.3, 0.5], 'thresholds': None},
        {'criterion': 'CrossEntropyLoss',
         'alpha': [[0.5, 0.25, 0.25], [0.2, 0.4, 0.4], [0.1, 0.1, 0.8]], 'thresholds': [15, 30]},
    ],
    'optimizer': ['Adam'],
    'lr': [1e-4, 5e-4],
    'epochs': [50],
    'scheduler': [
        {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1},
        {'type': 'CosineAnnealingLR', 'T_max': 50},
    ],
    'early_stopping': [
        {'patience': 15, 'delta': 0.0005},
    ],
}


def accuracy_fn(y_true, y_pred):

    """
    Percentage of correctly classified samples, ignoring samples with no label at this level (-1).
    gridsearch averages this score over the levels.
    """

    mask = y_true != -1
    if mask.sum() == 0:
        return 0.0
    correct = torch.eq(y_true[mask], y_pred[mask]).sum().item()
    return (correct / mask.sum().item()) * 100


def update_tuning_record(**fields):

    """ Adds or overwrites fields in the tuning JSON. """

    with open(tuning_path) as f:
        record = json.load(f)
    record.update(fields)
    with open(tuning_path, 'w') as f:
        json.dump(record, f, indent = 2)


if RUN_GRIDSEARCH:

    model = BCNN_Model(
        weights_directory = weights_directory,
        dim_outputs = dim_outputs,
        levels = LEVELS,
        device = device,
        model_name = MODEL_NAME,
        seed = SEED
    )

    BEST_HYPERPARAMETERS, BEST_SCORE = model.gridsearch(
        parameter_grid = HYPERPARAMETER_SEARCH_GRID,
        train_loader = train_loader,
        val_loader = val_loader,
        scoring_fn = accuracy_fn
    )

    tuning_name = f'TUNE_BCNN_ZOOP_{model.model_id}_{MODEL_NAME}_{LEVELS}'
    tuning_path = os.path.join(tuning_directory, tuning_name + '.json')

    with open(tuning_path, 'w') as f:
        json.dump({
            'model_name': MODEL_NAME,
            'levels': LEVELS,
            'seed': SEED,
            'max_class_size': MAX_CLASS_SIZE,
            'data_subdirectories': data_subdirectories,
            'batch_size': BATCH_SIZE,
            'tuning_split': {'train': TRAIN_PROP, 'val': VAL_PROP, 'test': TEST_PROP},
            'best_score': BEST_SCORE,
            'best_hyperparameters': BEST_HYPERPARAMETERS,
            'search_grid': HYPERPARAMETER_SEARCH_GRID,
        }, f, indent = 2)
    print(f'Tuning results saved to {tuning_path}')

    del model

else:

    tuning_path = os.path.join(tuning_directory, TUNING_JSON)
    tuning_name = TUNING_JSON.removesuffix('.json')
    with open(tuning_path) as f:
        tuning_record = json.load(f)
    BEST_HYPERPARAMETERS = tuning_record['best_hyperparameters']
    BEST_SCORE = tuning_record['best_score']
    print(f'Loaded best hyperparameters from {tuning_path}')

    # Warn if this script's setup differs from the one used for tuning
    current = {'model_name': MODEL_NAME, 'levels': LEVELS, 'seed': SEED,
               'max_class_size': MAX_CLASS_SIZE, 'batch_size': BATCH_SIZE,
               'data_subdirectories': data_subdirectories}
    for key, value in current.items():
        if key in tuning_record and tuning_record[key] != value:
            print(f'WARNING: {key} = {value} here, but tuning used {tuning_record[key]}')

print(f'\nBest validation score (mean accuracy over levels): {BEST_SCORE:.2f}')
print(f'Best hyperparameters: {BEST_HYPERPARAMETERS}')


# ################################################################################
#           RETRAIN WITH THE BEST HYPERPARAMETERS & SAVE
# ################################################################################

# Two retrainings, each with its own random split (same SEED):
#   'evaluation' : same 70/10/20 split as tuning -> unbiased test metrics 
#   'production' : 90/05/05 split -> this is the model to deploy.
#                  
FINAL_MODELS = {
    'evaluation': {'train': TRAIN_PROP, 'val': VAL_PROP, 'test': TEST_PROP, 'suffix': '_tuned'},
    'production': {'train': 0.90, 'val': 0.05, 'test': 0.05, 'suffix': '_production'},
}
TRAIN_MODELS = ['evaluation', 'production']   # remove one to skip it

SAVE = True


def retrain_and_save(kind, split):

    """ Retrains the BCNN with BEST_HYPERPARAMETERS on the given split and saves it. """

    print(f'\n==== Retraining {kind} model | split train/val/test = '
          f"{split['train']}/{split['val']}/{split['test']} ====")

    # val must be > 0: training computes the validation loss every epoch (and early stopping uses it)
    if split['val'] <= 0:
        raise ValueError(f'{kind}: the validation proportion must be > 0')

    train_idx, val_idx, test_idx = hier_dataset.split_train_test_val(
        train_prop = split['train'], val_prop = split['val'], test_prop = split['test']
    )
    train_dl, val_dl, test_dl = hier_dataset.create_dataloaders(
        batch_size = BATCH_SIZE,
        train_indices = train_idx,
        val_indices = val_idx,
        test_indices = test_idx,
        image_transforms = None,
        train_sample_weights = None
    )

    hyperparameters = dict(BEST_HYPERPARAMETERS)
    hyperparameters['batch_size'] = BATCH_SIZE

    model = BCNN_Model(
        weights_directory = weights_directory,
        dim_outputs = dim_outputs,
        levels = LEVELS,
        device = device,
        model_name = MODEL_NAME,
        seed = SEED
    )
    model.train(hyperparameters = hyperparameters, train_loader = train_dl, val_loader = val_dl)

    labels, probs, preds, logits = model.predict(test_loader = test_dl)
    test_scores = [accuracy_fn(labels[l].cpu(), preds[l].cpu()) for l in range(LEVELS)]
    print(f'{kind} test accuracy per level (coarsest first): ' + ', '.join(f'{s:.2f}' for s in test_scores))

    run_name = f"BCNN_ZOOP_{model.model_id}_{MODEL_NAME}_{LEVELS}{split['suffix']}"

    if SAVE:
        metadata = {
            'model_id': model.model_id,
            'model_name': MODEL_NAME,
            'run_name': run_name,
            'model_kind': kind,
            'dataset': hier_dataset,
            'classes': hier_dataset.class_names,
            'class_ids': hier_dataset.class_ids,
            'train_metrics': model.train_results,
            'test_loader': test_dl,
            'hyperparameters': hyperparameters,
            'image_transforms': hier_dataset.image_transforms,
            'max_class_size': MAX_CLASS_SIZE,
            'levels': LEVELS,
            'split': {k: split[k] for k in ('train', 'val', 'test')},
            'tuning': {'name': tuning_name, 'best_val_score': BEST_SCORE},
        }

        print(f'Saving weights, predictions, and metadata. Run Name: {run_name}')
        torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name + '.pth'))
        torch.save((labels, probs, preds, logits), os.path.join(results_directory, 'predictions', run_name + '.pth'))
        torch.save(metadata, os.path.join(results_directory, 'environment', run_name + '.pth'))

        # Link the tuning file to this model (read by bcnn_results_figures.ipynb)
        split_info = {k: split[k] for k in ('train', 'val', 'test')}
        if kind == 'evaluation':
            update_tuning_record(final_run_name = run_name, final_split = split_info)
        else:
            update_tuning_record(production_run_name = run_name, production_split = split_info)

    del model
    torch.cuda.empty_cache()
    return run_name


for kind in TRAIN_MODELS:
    retrain_and_save(kind, FINAL_MODELS[kind])
