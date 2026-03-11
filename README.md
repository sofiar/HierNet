
🛑 **ATTENTION**: This project is currently in progress.

# Hierarchical Model for image classification
This repository provides a Python library for trining and evaluating hierarchical CNN for image classification. It includes implementations of flat CNN models and branch CNN (BCNN) architectures that leverage hierarchical structures.

## 📂 Content 

**Library modules**: 
* `samples_setup.py`: Defines and constumizes the PyTorch `Dataset` and `HierDataset` classes
for loading and processing images for training.
* `model_setup.py`: Defines and constumizes the PyTorch `Model` for training and running inference on flat CNN models using DenseNet121 or ResNet50 architectures.
* `bcnn_setup.py`: Defines and costimizes the PyTorch `BCNN_Model` for training and runnig inference on Branch convolutional neural networks based on VGG16, ResNet50 or DenseNet121 architectures.
* `extra_funcitions.py`: Contains utility function for environment set-up.  

**Data analysis**:
For both datasets (Plankton-WHOI and Zooplankton-MNR)
* `summarize_data.ipynb`: Explores and visualizes the dataset.
* `train_baseline.py`:  Trains a baseline CNN with final-node class labels. 
* `train_bcnn.py`:  Trains a BCNN with 2 or 3 levels. 
* `get_baseline_results.ipynb`: Examines results from flat CNN models.
* `get_bcnn_results.ipynb`: Examines results from BCNN models.


## 🔧 Instalation and setup

1. If working on cluster or module system load python in your environment
```
module load python
```
2. Install `InformedML-CV` by : 
```
pip install git+https://github.com/sofiar/InformedML-CV.git
```
3. Install `hierclassifier` by : 
```
pip install git+https://github.com/sofiar/HierNet.git
```
4. Install any dependencies: Make sure required libraries are installed, including:
   * `torch`
   * `seaborn`
   * `sklearn`
   * `numpy`


## 📊 Data analysis
The hierarchical classification methods implemented in this library were applied to the following datasets:
### a. Plankton-WHOI
This dataset was built from a subset of the original dataset [WHOI-Plankton dataset (2006 to 2014)](https://darchive.mblwhoilibrary.org/collections/aad045e7-1fcf-5650-82ee-c62bd604d225). It contains marine plankton images. The hierarchical models were constructed following the hierarchy structure outlined below: 

<p align="center">
<img width="480" height="270" alt="Flow chart" src="https://github.com/user-attachments/assets/9b633f49-46eb-4e3d-aff7-598aa8cb225e" />
</p>

### b. Zooplankton-MNR
The dataset used here was provided by the Ministry of Natural Resources and Forestry (Ontario, Canada) and it contains pre-labeled images of fresha water zooplankton collected using FlowCam devides. The hierarchical models were constructed following the hierarchy structure outlined below

<p align="center">
<img width="480" height="270" alt="Flow chart" src="https://github.com/user-attachments/assets/6182850d-0cb8-444f-b0a5-bc0bdfe882b8"  />
</p>

## 📄 References
Zhu, X., & Bain, M. (2017). B-CNN: Branch Convolutional Neural Network for Hierarchical Classification. ArXiv, abs/1709.09890.


