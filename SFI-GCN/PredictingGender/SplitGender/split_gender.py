import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

fc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
sc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'

gender = '/home/ananya012/MainCodebase/SFI-GCN/scores/Gender_HCP.npy'

fc = np.load(fc)
sc = np.load(sc)
gender = np.load(gender)

# print shapes of data
print(fc.shape, sc.shape, gender.shape)


num_folds = 5
stratifiedKFolds = StratifiedKFold(n_splits=num_folds, shuffle=True)

path = '/home/ananya012/MainCodebase/SFI-GCN/PredictingGender/SplitGender/presplit5/'

if not os.path.exists(path):
    os.makedirs(path)
# save split indices
split_indices = {'train': [], 'test': []}
for fold, (train_idx, test_idx) in enumerate(stratifiedKFolds.split(fc, gender)):
    split_indices['train'].append(train_idx)
    split_indices['test'].append(test_idx)
    np.save(path+f'train_fold_{fold+1}.npy', train_idx)
    np.save(path+f'test_fold_{fold+1}.npy', test_idx)