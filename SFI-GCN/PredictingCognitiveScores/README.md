# Predicting Cognitive Scores

This file contains code for predicting cognitive scores using SFI-GCN (our model), MV-GCN, GCN and Linear Regression.

## Project Structure

- `lin_reg.py`: Script for linear regression using FC and SC data.
- `model.py`: Contains the models used for prediction (GCN, SC_FC_Inter_GCN, MV_GCN).
- `train_fusion.py`: Script for training and evaluating the main model SC_FC_Inter_GCN using a fusion of FC and SC data.
- `train_fusion_GCN_FC.py`: Script for the ablation study using only FC data with GCN.
- `train_fusion_GCN_SC.py`: Script for the ablation study using only SC data with GCN.
- `train_MV_GCN.py`: Script for comparison with MV-GCN model.
- `utils.py`: Functions for creating the dataset.

## Usage

To train the models and perform regression tasks for cognitive scores, use the following commands:

### SFI-GCN
- Run 
```bash 
python train_fusion.py
```

### MV-GCN
- Run 
```bash 
python train_MV_GCN.py
```

### Linear Regression
- Run 
```bash 
python lin_reg.py
```

### Ablation studies using GCN
- Run 
```bash 
python train_fusion_GCN_FC.py
python train_fusion_GCN_SC.py`
```

### For all models:
- Upon execution, you will be prompted to select the cognitive score to predict: <br />
`0: CogFluidComp` <br />                                                                                                             
`1: PicSeq`<br />
`2: PicVocab`<br />
`3: ReadEng`<br />
`4: CardSort`<br />
`5: ListSort`<br />
`6: Flanker`<br />
`7: ProcSpeed`<br />
`Enter the index of the score you want to predict:`<br />
- Enter desired index
- Results will be outputted to specified directories

## Data directory
- FC dataset: Located in /HCP_FC/X.npy
- SC dataset: Located in /HCP_SC/X.npy
- Scores: Located in /scores, one for each score

## Results directory
- Edit `data_dir` in the codes to alter where results are saved
