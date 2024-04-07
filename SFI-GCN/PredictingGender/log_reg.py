# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.pipeline import make_pipeline


fconnpath = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
sconnpath = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'

all_data_fc = np.load(fconnpath)
all_data_sc = np.load(sconnpath)
all_labels = np.load('/home/ananya012/MainCodebase/SFI-GCN/scores/Gender_HCP.npy')

# print all sizes in a line
print(all_data_fc.shape, all_data_sc.shape, all_labels.shape)

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
all_data_fc = corr_mx_flatten(all_data_fc)
all_data_sc = corr_mx_flatten(all_data_sc)
all_data = np.concatenate((all_data_fc, all_data_sc), axis=1) # horizontally

# print shapes
print("final shapes:", all_data.shape, all_labels.shape)

scaler = StandardScaler()

print("starting log reg")
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1
for train_index, test_index in skf.split(all_data, all_labels):
    print(f"in fold {fold}")
    fold+=1
    X_train, X_test = all_data[train_index], all_data[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]
    
    # only scales the features 
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    accuracies.append(accuracy)
    print("Accuracy:", accuracy)    

average_accuracy = np.mean(accuracies)
print(f"sverage accuracy: {average_accuracy:.4f}")
