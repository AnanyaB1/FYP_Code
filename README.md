# Predicting Gender and Cognitive Scores from Structural and Functional Connectome

## Porject Structure Overview

- `HCP_FC` folder: Contains Functional Connectome X.npy dataset for 836 subjects.
- `HCP_SC` folder: Contains Structural Connectome X.npy dataset for 836 subjects.
- `HCP_Transformer` folder: Contains the Transformer Model for the prediction of cognitive scores, and utility scripts for data preparation, training, testing, and result aggregation.
- `SFI-GCN` folder: Contains the SFI-GCN Models for the prediction of cognitive scores and gender, and utility scripts for gata preparation, training, testing, and result aggregation.
- `scores` folder: Contains .npy files for cognitive scores and gender.

## Detailed Project Structure
```plaintext
.
├── HCP_FC/
│   └── X.npy
├── HCP_SC/
│   └── X.npy
├── HCP_Transformer/
│   ├── create_Y_traintest.py
│   ├── main_trans.py
│   ├── overall_results.py
│   ├── README.md
│   ├── train_test_functs.py
│   ├── transformer_model.py
│   └── utils.py
├── SFI-GCN/
│   ├── Generating_Results_From_Runs/
│   │   ├── README.md
│   │   ├── results_overall.py
│   │   ├── results_overall_ablation.py
│   │   └── results_overall_ablation_gender.py
│   ├── Generating_Y_Datasets/
│   │   ├── extract_scores.py
│   │   ├── HCP_metadata.csv
│   │   ├── overlap_HCP.txt
│   │   └── README.md
│   ├── PredictingCognitiveScores/
│   │   ├── lin_reg.py
│   │   ├── model.py
│   │   ├── README.md
│   │   ├── train_fusion.py
│   │   ├── train_fusion_GCN_FC.py
│   │   ├── train_fusion_GCN_SC.py
│   │   ├── train_MV_GCN.py
│   │   └── utils.py
│   ├── PredictingGender/
│   │   ├── log_reg.py
│   │   ├── model.py
│   │   ├── README.md
│   │   ├── SplitGender/
│   │   │   └ ... (presplit data)
│   │   ├── svc.py
│   │   ├── train_fusion_class.py
│   │   ├── train_fusion_GCN_FC_Class.py
│   │   ├── train_fusion_GCN_SC_Class.py
│   │   └── utils.py
│   └── README.md
└── scores/
    ├── CardSort_Unadj_HCP.npy
    ├── CogFluidComp_Unadj_HCP.npy
    ├── Flanker_Unadj_HCP.npy
    ├── Gender_HCP.npy
    ├── ListSort_Unadj_HCP.npy
    ├── PicSeq_Unadj_HCP.npy
    ├── PicVocab_Unadj_HCP.npy
    ├── ProcSpeed_Unadj_HCP.npy
    └── ReadEng_Unadj_HCP.npy
```
## Data directory
- FC dataset: Located in /HCP_FC/X.npy
- SC dataset: Located in /HCP_SC/X.npy
- Scores and gender: Located as .npy files in /scores