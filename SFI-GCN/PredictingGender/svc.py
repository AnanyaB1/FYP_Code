from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
import numpy as np
from sklearn.pipeline import make_pipeline
import pandas as pd
import time
import os



x_fc = f'C:/Users/anany/OneDrive - Nanyang Technological University/Desktop/FYP/gae/HCP/X_FC.npy'  
x_sc = f'C:/Users/anany/OneDrive - Nanyang Technological University/Desktop/FYP/gae/HCP_SC/X_SC.npy' 
y_score = f'C:/Users/anany/OneDrive - Nanyang Technological University/Desktop/FYP/scores/Gender_HCP.npy' 

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

print(y_score)
print("finished loading data")
kf = KFold(n_splits=5, shuffle=True)
pipeline = make_pipeline(StandardScaler(), SVC())
print("pipeline finish")

accuracies, precisions, recalls, f1_scores = [], [], [], []
fold = 1
for train_index, test_index in kf.split(all_data, y_score):
    print("in fold ", fold)
    fold += 1
    X_train, X_test = all_data[train_index], all_data[test_index]
    y_train, y_test = y_score[train_index], y_score[test_index]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Print average metrics
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

data = {
    'Average Accuracy': [np.mean(accuracies)],
    'Std Dev Accuracy': [np.std(accuracies)],
    'Average Precision': [np.mean(precisions)],
    'Std Dev Precision': [np.std(precisions)],
    'Average Recall': [np.mean(recalls)],
    'Std Dev Recall': [np.std(recalls)],
    'Average F1 Score': [np.mean(f1_scores)],
    'Std Dev F1 Score': [np.std(f1_scores)]
}


path = f'C:/Users/anany/OneDrive - Nanyang Technological University/Desktop/FYP/SVC/Gender'
# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)
if not os.path.exists(path):
    os.makedirs(path)
final_dest = path + "/" + str(int(time.time())) + 'svc_metrics.csv'
df.to_csv(final_dest, index=False)