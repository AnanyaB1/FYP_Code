from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline



# load data
scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]  # Removed duplicate "CardSort"
for i in range(len(scores)):
    print(f"{i}: {scores[i]}")

index = int(input("Enter the index of the score you want to predict: "))
score = scores[index]

x_fc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
x_sc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'
y_score = f'/home/ananya012/MainCodebase/SFI-GCN/scores/{score}_Unadj_HCP.npy'

x_fc = np.load(x_fc)
x_sc = np.load(x_sc)
y_score = np.load(y_score)


def corr_mx_flatten(X):
    """
    Takes in a list of matrices, flattens them and returns a Numpy array of flattened matrices
    Inputs:
    - X: List or Numpy array of matrices
    Return:
    - X_flattened: Numpy array of flattened matrices
    """

    num_features = int((X.shape[1]) * (X.shape[1] - 1) * 0.5)
    X_flattened = np.empty((X.shape[0], num_features))

    for i, matrix in enumerate(X):
        matrix_upper_triangular = matrix[np.triu_indices(np.shape(matrix)[0],1)]
        X_flattened[i] = np.ravel(matrix_upper_triangular, order="C")

    return X_flattened

# take only the upper triangle of fc and sc
x_fc = corr_mx_flatten(x_fc)
x_sc = corr_mx_flatten(x_sc)
all_data = np.concatenate((x_fc, x_sc), axis=1) # horizontally


print("finished loading data")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), LinearRegression())
print("pipeline finish")
corrs, rmses, maes, r2s = [], [], [], []
fold = 1
for train_index, test_index in kf.split(all_data, y_score):
    print("in fold ", fold)
    fold+=1
    X_train, X_test = all_data[train_index], all_data[test_index]
    y_train, y_test = y_score[train_index], y_score[test_index]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    corrs.append(corr)
    rmses.append(rmse)
    maes.append(mae)
    r2s.append(r2)


# Print average metrics
print(f"Average RMSE: {np.mean(rmses):.4f}")
print(f"Average MAE: {np.mean(maes):.4f}")
print(f"Average Correlation: {np.mean(corrs):.4f}")
print(f"Average R^2: {np.mean(r2s):.4f}")

data = {
    'Average RMSE': [np.mean(rmses)],
    'Std Dev RMSE': [np.std(rmses)],
    'Average MAE': [np.mean(maes)],
    'Std Dev MAE': [np.std(maes)],
    'Average Correlation': [np.mean(corrs)],
    'Std Dev Correlation': [np.std(corrs)],
    'Average R^2': [np.mean(r2s)],
    'Std Dev R^2': [np.std(r2s)]
}

import pandas as pd
import time

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)
import os
if not os.path.exists(f'/data/ananya012/results/Linear_Regression/{score}'):
    os.makedirs(f'/data/ananya012/results/Linear_Regression/{score}')

data_dir = f'/data/ananya012/results/Linear_Regression/{score}/' + str(int(time.time())) + '-metrics.csv'
df.to_csv(data_dir, index=False)