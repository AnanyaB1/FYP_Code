import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformer_model import TransformerRegressor, Block, Transformer, MLPBlock
from train_test_functs import train_recon, test_recon, train_predict, test_predict, test_predict_and_save
from utils import bin_y, StandardScaler, CustomDataset



####################################################load in data#################################################################
y_dir = '/home/ananya012/SC-FC-fusion/gae/scores/'

scfc_dir = '/home/ananya012/HCP_Transformer/Data/'
scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]

x_fc = np.load(scfc_dir+"X_FC.npy")

x_sc = np.load(scfc_dir+"X_SC.npy")

for i in range(len(scores)):
    print(f"{i}: {scores[i]}")

index = int(input("Enter the index of the score you want to predict: "))
score = scores[index]

y = np.load(y_dir + f"{score}_Unadj_HCP.npy")

print("shapes of fc, sc and y:", x_fc.shape, x_sc.shape, y.shape)
print("finished loading data")

#############################################################files to save##############################################################
parent_folder = f'/home/ananya012/HCP_Transformer/Data/Train_Test_Only/{score}'
    # check if folder if not make
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)

home_dir = f'/home/ananya012/HCP_Transformer/Data/Train_Test_Only/{score}/'+'Trans_'+str(int(time.time()))+'/'
os.makedirs(home_dir)


##############################################################model run################################################################
y_binned = bin_y(y)

# Parameters
input_dim = 232  # dimension of the input features
model_dim = 128  # dimensionality of the model
num_heads = 4   # number of attention heads
num_layers = 1  # number of transformer layers
output_dim = 1  # output dimensionality
num_splits = 5 # number of splits

kf = StratifiedKFold(n_splits=num_splits, shuffle=True)

overall_corrs = []
for fold, (train_idx, test_idx) in enumerate(kf.split(x_fc, y_binned)):
    print("in fold", fold+1)
    # split based on indexes
    x_fc_train, x_fc_test = x_fc[train_idx], x_fc[test_idx]
    x_sc_train, x_sc_test = x_sc[train_idx], x_sc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # val_test_ratio = 0.5
    # x_fc_test, x_fc_val, x_sc_test, x_sc_val, y_test, y_val = train_test_split(x_fc_test, x_sc_test, y_test, test_size=val_test_ratio, stratify=bin_y(y_test))

    # standardise
    ss_fc = StandardScaler()
    ss_sc = StandardScaler()
    ss_y = StandardScaler()

    x_fc_train = ss_fc.fit_transform(x_fc_train)
    x_fc_test = ss_fc.transform(x_fc_test)
    # x_fc_val = ss_fc.transform(x_fc_val)

    x_sc_train = ss_sc.fit_transform(x_sc_train)
    x_sc_test = ss_sc.transform(x_sc_test)
    # x_sc_val = ss_sc.transform(x_sc_val)

    y_train = (ss_y.fit_transform(y_train.reshape(-1, 1))).flatten()
    y_test = (ss_y.transform(y_test.reshape(-1, 1))).flatten()
    # y_val = (ss_y.transform(y_val.reshape(-1, 1))).flatten()

    # concat
    x_train = np.concatenate((x_fc_train, x_sc_train), axis=-1)
    x_test = np.concatenate((x_fc_test, x_sc_test), axis=-1)
    # x_val = np.concatenate((x_fc_val, x_sc_val), axis=-1)

    # dataset
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    # val_dataset = CustomDataset(x_val, y_val)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Create the model
    model = TransformerRegressor(input_dim, model_dim, num_heads, num_layers, output_dim)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #, weight_decay=0.01)
    criteria = nn.MSELoss() #nn.MSELoss()
    recreation_criteria = nn.MSELoss()

    # reconstruction training
    # 100, 150 too less
    ######################################################reconstruction phase##################################################################
    for epoch in range(250):
        train_loss = train_recon(model, device, train_loader, optimizer, criteria)

        # evaluate every n steps
        if not epoch % 10:
            test_loss, test_corr = test_recon(model, device, test_loader, criteria)
            train_loss, train_corr = test_recon(model, device, train_loader, criteria)
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Corr: {train_corr}, Test Corr: {test_corr}")
        # print(f"Batch: {epoch}, Train Loss: {train_loss}")

    ######################################################prediction phase##################################################################
    # prediction training
    best_corr = 0
    cur_epoch = 0
    cur_train_corr = 0
    patience = 100
    early_stop = False
    wait = 0
    
    model_save_path = home_dir+f'best_model_fold_{fold+1}.pth'
    for epoch in range(500):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        train_loss = train_predict(model, device, train_loader, optimizer, criteria)

        test_loss, test_corr = test_predict(model, device, test_loader, criteria)
        train_loss, train_corr = test_predict(model, device, train_loader, criteria)

        if test_corr > best_corr:
            best_corr = test_corr
            wait = 0  # Reset wait time since we have improvement
            cur_train_corr = train_corr
            cur_epoch = epoch
            print(f"Epoch {epoch}: Improvement found, test_corr: {test_corr}")
            # Save model if this is your best model
            torch.save(model.state_dict(), model_save_path)
        else:
            wait += 1
            if wait >= patience:
                early_stop = True  # Trigger early stopping
                print(f"No improvement for {patience} consecutive epochs, stopping early.")
            # print(f"Epoch: {epoch}, Test Corr: {test_corr}, waiting for improvement: {wait} epochs.")
        
        if not epoch % 10:
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Corr: {train_corr}, Test Corr: {test_corr}")

    
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_corr, test_rmse, test_mae, test_r2, y_true, y_pred = test_predict_and_save(model, device, test_loader, criteria, ss_y)
    # val_loss, val_corr, val_rmse, val_mae, val_r2, y_true, y_pred = test_predict_and_save(model, device, val_loader, criteria, ss_y)
    train_loss, train_corr = test_predict(model, device, train_loader, criteria)
    print(f"Prediction of {score}")

    print(f"Train Loss: {train_loss}, Train Corr: {train_corr}")
    print(f"Test Loss: {test_loss}, Test Corr: {test_corr}, Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")
    # print(f"Val Loss: {val_loss}, Val Corr: {val_corr}, Val RMSE: {val_rmse}, Val MAE: {val_mae}, Val R2: {val_r2}")
    print()

    overall_corrs.append(test_corr)

    df_test_predict = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    test_pred_file = home_dir+f'test_prediction_fold_{fold+1}.csv'
    df_test_predict.to_csv(test_pred_file, index=False)

    # save the test corr, rmse, mae and r2 as final results for this fold
    metrics_results_file = home_dir+f'metrics_result_fold_{fold+1}.csv'
    df_metrics_results = pd.DataFrame({
        'Test Corr': [test_corr],
        'Test RMSE': [test_rmse],
        'Test MAE': [test_mae],
        'Test R2': [test_r2]
    })
    df_metrics_results.to_csv(metrics_results_file, index=False)


print(overall_corrs)
print(np.mean(overall_corrs))
# std
