import os
import torch
from torchvision import transforms
from samples_setup import ImageDataset, merge_classes
from model_setup import Model
from extra_functions import set_seed

########################### Environment set up #################################

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Specify paths
data_directory = '/data/WHOI-Plankton'
data_subdirectories = ['2010','2011', '2012','2013','2014']

# Specify other environment variables
SEED = 666
set_seed(SEED)

############################# Data preparation #################################

# Set classes
PLANKTON_CLASSES = [
    'Cylindrotheca',
    'Cerataulina',
    'Chaetoceros',
    'Chaetoceros_didymus_flagellate',
    'Corethron',
    'Dactyliosolen',
    'Ditylum',
    'Eucampia',
    'Guinardia_delicatula',
    'Guinardia_striata',
    'G_delicatula_external_parasite',
    'Leptocylindrus',
    'Pseudonitzschia',
    'Rhizosolenia',
    'Skeletonema',
    'Thalassiosira',
    'Thalassionema'         
    ]

# Create ImageDataset
MAX_CLASS_SIZE = 10000

dataset_selected = ImageDataset(
    data_directory = data_directory,
    data_subdirectories = data_subdirectories,
    class_names = PLANKTON_CLASSES,
    max_class_size = MAX_CLASS_SIZE,
    image_resolution = 64,
    image_transforms = None,
    format_file = '.png',
    seed = SEED
    )

# Merge categories
classes_to_merge_list = [
    [
    'Guinardia_delicatula',
    'Guinardia_striata',
    'G_delicatula_external_parasite',
   ],
   [
    'Chaetoceros',
    'Chaetoceros_didymus_flagellate',   
   ]
]
new_names_list = [
    'Guinardia',
    'Chaetoceros'
]

# Define final dataset
dataset =  merge_classes(
    dataset = dataset_selected,
    classes_to_merge_list=classes_to_merge_list,
    new_names_list=new_names_list
    )

NUM_CLASSES = len(dataset.class_ids)

##################### Add Image Transformations to Pipeline ####################

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding = 5, fill = 0),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


dataset.append_image_transforms(
    image_transforms = train_transforms, verbose = False, replace = True
)

dataset.print_image_transforms()

################## Split data into train, test and validation ##################
 
TRAIN_PROP = 0.7
VAL_PROP = 0.1
TEST_PROP = 0.2

BATCH_SIZE = 64

train_split, val_split, test_split = dataset.split_train_test_val(
    train_prop = TRAIN_PROP, val_prop = VAL_PROP, test_prop = TEST_PROP, verbose = False
)

############################ Create data loaders ###############################

train_loader, val_loader, test_loader = dataset.create_dataloaders(
    batch_size = BATCH_SIZE,
    train_indices = train_split,
    val_indices = val_split,
    test_indices = test_split,
    image_transforms = None,
    train_sample_weights = None
)

############################# Train model ######################################

# Define model 
MODEL_NAME = 'densenet121'
weights_directory = '/data/zooplankton_data'

model = Model(
    weights_directory = weights_directory,
    device = device,
    num_classes = NUM_CLASSES,
    model_name = MODEL_NAME
)

# Define hyperparamters
HYPERPARAMETERS = {
    'loss_fn': {'type': 'CrossEntropyLoss', 'weights': None}, 
    'optimizer': 'Adam', 
    'lr': 5e-4, 
    'epochs': 50, 
    'scheduler':{'type': 'CosineAnnealingLR', 'T_max': 50},
    'early_stopping': {'patience': 15, 'delta': 0.005}
}


model.train(
    hyperparameters = HYPERPARAMETERS,
    train_loader = train_loader,
    val_loader = val_loader
)


################################# Inference ####################################

labels, probs, preds, logits = model.predict(test_loader = test_loader)

############################### Save Results ###################################

MODEL_ID = model.model_id
SUFFIX = '' # UPDATE FOR CUSTOM SUFFIX
run_name = f'{MODEL_ID}_{MODEL_NAME}{SUFFIX}'
results_directory = '/home/ruizsuar/Plankton-h-classifier/Models_results'

metadata = {
    'model_id': MODEL_ID,
    'model_name': MODEL_NAME,
    'run_name': run_name,
    'dataset': dataset,
    'classes': dataset.class_names,
    'class_map': {name: idx for name, idx in zip(dataset.class_names, dataset.class_ids)},
    'train_metrics': model.train_results,
    'test_loader': test_loader,
    'hyperparameters': HYPERPARAMETERS,
    'image_transforms': dataset.image_transforms,
    'max_class_size': MAX_CLASS_SIZE,
}

SAVE = True

if SAVE:
    print(f'Saving weights, predictions, and metadata. Model: {MODEL_NAME} (ID: {MODEL_ID})')
    print(f'Run Name: {run_name}')

    # Save learned weights, predictions and results
    torch.save(model.model.state_dict(), os.path.join(results_directory, 'weights', run_name + '.pth'))
    torch.save((labels, probs, preds, logits), os.path.join(results_directory, 'predictions', run_name + '.pth'))
    torch.save(metadata, os.path.join(results_directory, 'environment', run_name + '.pth'))

# Delete model objects
del model




