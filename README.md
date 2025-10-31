# Hierarchical Model for Plankton Classification
This repository contains scripts to train and test different hierarchical CNN approaches for classifing images of plankton. 

### 📊 Data
The dataset used is the [WHOI-Plankton dataset (2006 to 2014)](https://darchive.mblwhoilibrary.org/collections/aad045e7-1fcf-5650-82ee-c62bd604d225)

 ### 📂 Content 
 * `samples_setup.py`: Defines and constume PyTorch `Dataset` for loading and processing images for training.
 * `model_setup.py`: Defines and constume PyTorch `Model` for training and running inference using DenseNet121 or ResNet50 architectures. 
 * `extra_funcitions.py`: Contains utility function for environment set-up.  
 * `train_baseline.py`:  Trains a baseline CNN with final nodes class labels. 
