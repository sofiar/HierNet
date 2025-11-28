import os
import torch
from torchvision import transforms
from samples_setup import ImageDataset, HierImageDataset, merge_classes
from model_setup import Model
from extra_functions import set_seed
from bcnn_setup import BCNN_Model, BcnnVGG

########################### Environment set up #################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Specify paths
data_directory = '/data/Hier-Zooplankton'
data_subdirectories = ['IssacData']

# Specify other environment variables
SEED = 666
set_seed(SEED)

############################# Data preparation #################################

ZOOPLANKTON_CLASSES = [
     'Debris',
     'Bubbles',
     'Copepoda',
     'Calanoid',
     'Cyclopoid',
     'Harpacticoid',
     'Cladocera',
     'Bosminidae',
     'Daphnia',
]

# 1. Base dataset

MAX_CLASS_SIZE = 10000
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

# 2.Create hierarchical dataset

LEVELS = 3

coarse_names1 = ['Zoop-yes', 'Zoop-No']
groups1 = [
    [
        'Copepoda','Cladocera','Bosminidae','Daphnia',
        'Cyclopoid','Harpacticoid','Calanoid'
    ],
    ['Debris','Bubbles']
]

coarse_names2 = ['Copepoda', 'Cladocera','Debris','Bubbles']
groups2 = [
    ['Copepoda','Cyclopoid','Calanoid','Harpacticoid'],
    ['Cladocera','Bosminidae','Daphnia'],
    ['Debris'],
    ['Bubbles']
]

coarse_names3 = [
    'Cyclopoid','Calanoid','Harpacticoid',
    'Bosminidae','Daphnia','Debris','Bubbles'
]
groups3 = [
    ['Cyclopoid'],
    ['Calanoid'],
    ['Harpacticoid'],
    ['Bosminidae'],
    ['Daphnia'],
    ['Debris'],
    ['Bubbles'] 
] 
coarse_names = [coarse_names3,coarse_names2,coarse_names1]
groups = [groups3, groups2, groups1]

hier_dataset = HierImageDataset(
    base_dataset=dataset,
    groups=groups,
    
    coarse_names = coarse_names
)
hier_dataset.print_dataset_details()

dim_outputs = []
for element in coarse_names:
    dim_outputs.insert(0,len(element))

##################### Add Image Transformations to Pipeline ####################

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
])


hier_dataset.append_image_transforms(
    image_transforms = train_transforms, replace = True
)

################## Split data into train, test and validation ##################
 
TRAIN_PROP = 0.7
VAL_PROP = 0.1
TEST_PROP = 0.2

BATCH_SIZE = 64

train_split, val_split, test_split = hier_dataset.split_train_test_val(
    train_prop = TRAIN_PROP, val_prop = VAL_PROP, test_prop = TEST_PROP
)

############################ Create data loaders ###############################

train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size = BATCH_SIZE,
    train_indices = train_split,
    val_indices = val_split,
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None
)

# ########################## Define and train model ##############################

MODEL_NAME = 'densenet121' 

model = BCNN_Model(
    weights_directory = './pre_trained_weights',
    dim_outputs = dim_outputs,
    levels = LEVELS,
    device = device,
    model_name = MODEL_NAME
)

# ########################## Hyperparameter tuning ##############################

# Specify Parameter Search Grid 
TUNE = False

HYPERPARAMETER_SEARCH_GRID = {
    'loss_fn': [
        {'criterion': 'CrossEntropyLoss', 'alpha': [1/3,1/3,1/3],'thresholds': None},
        {'criterion': 'CrossEntropyLoss', 'alpha': [[1/3,1/3,1/3],[0.2,0.2,0.6]],'thresholds':[20]},
    ],  
    'optimizer': ['Adam'],
    'lr': [1e-4,5e-4],
    'epochs': [50],
    'scheduler': [
        {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1},
        {'type': 'CosineAnnealingLR', 'T_max': 50},
    ],
    'early_stopping': [
        {'patience': 20, 'delta': 0.0005},
    ],
}


def accuracy_fn(y_true, y_pred):
    
    """ Defines accuracy measure as the percentage of samples well classified"""
    
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


if TUNE:
    
    HYPERPARAMETERS,_ = model.gridsearch(
        parameter_grid = HYPERPARAMETER_SEARCH_GRID,
        train_loader = train_loader,
        val_loader = val_loader,
        scoring_fn = accuracy_fn 
    )
else: 
    
    HYPERPARAMETERS = {
        'loss_fn': {
            'criterion': 'CrossEntropyLoss',
            'alpha':  [1/3,1/3,1/3],#[[0.5,0.25,0.25],[0.2,0.4,0.4],[0.1,0.1,0.8]],
            'thresholds': None#[15,20]
            }, 
        'optimizer': 'Adam', 
        'lr': 5e-4, 
        'epochs': 60, 
        'scheduler':{'type': 'CosineAnnealingLR', 'T_max': 50},
        'early_stopping': {'patience': 15, 'delta': 0.0005},
        #'early_stopping': None
    }
HYPERPARAMETERS['batch_size'] = BATCH_SIZE

model.train(
    hyperparameters = HYPERPARAMETERS,
    train_loader = train_loader,
    val_loader = val_loader
)

################################# Inference ####################################

labels, probs, preds, logits = model.predict(test_loader = test_loader)

############################### Save Results ###################################

MODEL_ID = model.model_id
MODEL_NAME = model.model_name
LEVELS = model.levels
run_name =  f"BCNN_ZOOP_{MODEL_ID}_{MODEL_NAME}_{LEVELS}"
results_directory = '/home/ruizsuar/Plankton-h-classifier/Models_results'

SAVE = True

metadata = {
    'model_id': MODEL_ID,
    'model_name': MODEL_NAME,
    'run_name': run_name,
    'dataset': hier_dataset,
    'classes': hier_dataset.class_names,
    'class_ids': hier_dataset.class_ids,
    'train_metrics': model.train_results,
    'test_loader': test_loader,
    'hyperparameters': HYPERPARAMETERS,
    'image_transforms': hier_dataset.image_transforms,
    'max_class_size': MAX_CLASS_SIZE,
    'levels': LEVELS
}


if SAVE:
    print(f'Saving weights, predictions, and metadata. Model: {MODEL_NAME} (ID: {MODEL_ID})')
    print(f'Run Name: {run_name}')

    # Save learned weights, predictions and results
    torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name + '.pth'))
    torch.save((labels, probs, preds, logits), os.path.join(results_directory, 'predictions', run_name + '.pth'))
    torch.save(metadata, os.path.join(results_directory, 'environment', run_name + '.pth'))

# Delete model objects
del model