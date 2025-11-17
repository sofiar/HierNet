# Hierarchical Model for Plankton Classification
This repository contains scripts to train and test different hierarchical CNN approaches for classifing images of marine plankton. It includes implementations of flat CNN models and branch CNN (BCNN) architectures that leverage hierarchical structures. 

## 📊 Data
The dataset used here is built from a subset of the original dataset [WHOI-Plankton dataset (2006 to 2014)](https://darchive.mblwhoilibrary.org/collections/aad045e7-1fcf-5650-82ee-c62bd604d225). The hierarchical models were constructed following the hierarchy structure outlined below: 

<p align="center">
<img width="720" height="405" alt="Flow chart" src="https://github.com/user-attachments/assets/9b633f49-46eb-4e3d-aff7-598aa8cb225e" />
</p>


## 📂 Content 

#### 🧩 Python scripts
* `samples_setup.py`: Defines and constumizes the PyTorch `Dataset` and `HierDataset` classes
for loading and processing images for training.
* `model_setup.py`: Defines and constumizes the PyTorch `Model` for training and running inference on flat CNN models using DenseNet121 or ResNet50 architectures.
* `bcnn_setup.py`: Defines and costimizes the PyTorch `BCNN_Model` for training and runnig inference on Branch convolutional neural networks based on VGG16, ResNet50 or DenseNet121 architectures.
* `extra_funcitions.py`: Contains utility function for environment set-up.  
* `train_baseline.py`:  Trains a baseline CNN with final-node class labels. 
* `train_bcnn.py`:  Trains a BCNN with 2 or 3 levels. 


#### 📘 Jupyter notebooks 
* `summarize_data.ipynb`: Explores and visualizes the dataset.
* `get_baseline_results.ipynb`: Examines results from flat CNN models.
* `get_bcnn_results.ipynb`: Examines results from BCNN models.


## 📄 References
Zhu, X., & Bain, M. (2017). B-CNN: Branch Convolutional Neural Network for Hierarchical Classification. ArXiv, abs/1709.09890.


