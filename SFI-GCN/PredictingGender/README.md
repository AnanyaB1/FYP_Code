# Predicting Gender

This folder contains code for predicting gender using SFI-GCN, GCN, Logisticc Regression and SVC.

## Project Structure

- `log_reg.py`: Logistic Regression model using combined SC and FC data
- `model.py`: Defines the models used for SFI-GCN and GCN.
- `svc.py`: SVC model using combined SC and FC data
- `train_fusion_class.py`: Main script for training and evaluating the main model SC_FC_Inter_GCN_Class model using fusion FC and SC data.
- `train_fusion_GCN_FC_Class.py`: For the ablation study with only FC data using GCN for gender classification.
- `train_fusion_GCN_SC_Class.py`: For the ablation study with only SC data using GCN for gender classification.
- `train_fusion_class_tuning.py`: Script for tuning hyperparameters of the gender classification model.
- `train_fusion_class_tuning_learningrate.py`: Hyperparameter tuning script focused on learning rate adjustments.
- `train_fusion_class_tuning_weights.py`: Hyperparameter tuning script focused on the class weights adjustments.
- `utils.py`: Utility functions supporting data handling and processing.
- SplitGender: Folder containing presplit gender data.

## Usage

To train the models and perform classification tasks, use the following commands:

### SFI-GCN-Class
- Run 
```bash 
python train_fusion_class.py
```

### Logistic Regression
- Run 
```bash 
python log_reg.py
```

### SVC
- Run `python svc.py`

## Ablation studies using GCN
- Run 
```bash 
python train_fusion_GCN_FC_Class.py
python train_fusion_GCN_SC_Class.py
```


### For all models:
- Results will be outputted to specified directories

## Data directory
- FC dataset: Located in /HCP_FC/X.npy
- SC dataset: Located in /HCP_SC/X.npy
- Gender: Located in /scores/Gender_HCP.npy