import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import KFold
from sklearn import preprocessing

from utils import create_dataset_SC
from torch_geometric.data import DataLoader
from model import GCN
import torch.nn
from torch.utils.data.dataloader import default_collate

#################### Functions ####################

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MAELoss(yhat,y):
    return torch.mean(torch.abs(yhat-y))

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

################### Load Data ####################
print("-----------------Loading Data---------------------")

fc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
sc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'

scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]

# choose which score to run model for
for i in range(len(scores)):
    print(f"{i}: {scores[i]}")

index = int(input("Enter the index of the score you want to predict: "))
score_name = scores[index]

score_path = f'/home/ananya012/MainCodebase/SFI-GCN/scores/{score_name}_Unadj_HCP.npy'

print(f"{score_name} chosen")

# load data 
fc = np.load(fc)
sc = np.load(sc)
score = np.load(score_path)

print("-----------------Data Loaded-------------------")

################## Parameter Setting ##################

# hidden layers
hidden_channels=115 #4 #64 #115
hc_gcn = hidden_channels*116
hc2 = 256 #128

# training
epoch_num = 19
decay_rate = 0.5  
decay_step = 19
lr =0.001 
num_folds = 5
batch_size = 5


#################### File Path ####################

# check if file exists, else create
base_dir = f'/data/ananya012/results/GCN/{score_name}/SC/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

data_dir = f'/data/ananya012/results/GCN/{score_name}/SC/GCN_SC-' + str(int(time.time()))
os.makedirs(data_dir)


param_file = data_dir + '/parameters.txt'
with open(param_file, 'w') as f:
    f.write(f"hidden_channels: {hidden_channels}\n")
    f.write(f"hc_gcn: {hc_gcn}\n")
    f.write(f"hc2: {hc2}\n")
    f.write(f"epoch_num: {epoch_num}\n")
    f.write(f"decay_rate: {decay_rate}\n")
    f.write(f"decay_step: {decay_step}\n")
    f.write(f"lr: {lr}\n")
    f.write(f"num_folds: {num_folds}\n")
    f.write(f"batch_size: {batch_size}\n")

#################### Train and Test Functions ####################

def train(model, optimizer, train_loader, train_label, device):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        
        # get output
        output = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)  # Perform a single forward pass.
        
        # get actual values 
        scores_actual = train_label[data.y]

        # calculate loss
        pre_loss = RMSELoss(output, torch.reshape(scores_actual.float(),output.shape))       
        loss =  pre_loss + L2Loss(model, 0.001)

        if loss == 'nan':
            break

        loss.backward()  # Deriving gradients.
        optimizer.step()  # Updating parameters based on gradients.
        optimizer.zero_grad()  # Clearing gradients.

def test(model, test_loader, test_label, device):
    model.eval()

    MAE_sum = 0
    rmse_sum = 0
    MAE_sum = torch.tensor(MAE_sum).to(device)
    rmse_sum = torch.tensor(rmse_sum).to(device)
    predictions = torch.tensor(()).to(device)
    target_scores = torch.tensor(()).to(device)
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        
        # output
        output = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)
        
        # actual values
        scores_actual = test_label[data.y]
        
        # metrics calculation
        rmse = RMSELoss(output, torch.reshape(scores_actual.float(), output.shape))
        mae = MAELoss(output, torch.reshape(scores_actual.float(), output.shape))
        
        # add to sum
        MAE_sum = MAE_sum + mae  
        rmse_sum = rmse_sum + rmse

        # append predicted, target, attention
        predictions = torch.cat((predictions, output), dim=0)    
        target_scores = torch.cat((target_scores, torch.reshape(scores_actual.float(), output.shape)), dim=0)
    
    # reshape target scores jic 
    target_scores = torch.reshape(target_scores,(len(predictions),1))
    
    # to cpu
    predictions = np.squeeze(np.asarray(predictions.cpu().detach().numpy()))
    target_scores = np.squeeze(np.asarray(target_scores.cpu().detach().numpy()))   
    
    # calculate metrics
    corr = scipy.stats.pearsonr(predictions, target_scores)[0]  
    rmse = rmse_sum/len(test_loader)
    mae = MAE_sum/len(test_loader)

    return corr,  rmse, mae, predictions, target_scores

#################### Main Code ####################
finalfile = data_dir +'/final_result.csv'        

print(score.shape)
num = sc.shape[0]
print("num", num)

fold = 0
kf = KFold(n_splits=num_folds, shuffle=True)

# data to save
true_out = np.squeeze(np.array([[]]))
pred_out = np.squeeze(np.array([[]]))
test_att_all = np.empty((1,116))

# metrics to save
test_corrs = []
test_rmses = []
test_maes = []

for X_train, X_test in kf.split(list(range(1,num))):
    fold = fold+1
    model = GCN(hidden_channels, hc_gcn, hc2)
    print("Model:\n\t",model)

    # cuda
    print(torch.cuda.is_available())
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"SC GCN on {score_name}")

    model.to(device)

    print(f"Fold: {fold}")

    
    train_data = sc[X_train]
    test_data = sc[X_test]

    train_score = score[X_train]
    test_score = score[X_test]

    # creates indices up to length of train and test set, then reshapes it to (len, 1)
    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))
    
    # create datasets
    training_dataset = create_dataset_SC(train_data, index_train)
    testing_dataset = create_dataset_SC(test_data, index_test)

    # conver to tensor and push scores to device
    train_score_input = torch.tensor(train_score).to(device)
    test_score_input = torch.tensor(test_score).to(device)

    # dataloader
    train_loader = DataLoader(training_dataset, batch_size, shuffle = True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(testing_dataset, batch_size, shuffle= True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr)

    ################define the files for each fold#############################################
    predict_test_csv = data_dir + f'/predict_test{fold}.csv'
    predict_train_csv = data_dir + f'/predict_train{fold}.csv'
    prediction_all_folds_csv = data_dir + '/prediction_all_folds.csv'
    performance_file = data_dir+'/performance_test'+str(fold)+'.csv' 
    df_name = {'epoch','train_rmse','train_mae','train_corr','test_rmse','test_mae','test_corr'}
    df_name = pd.DataFrame(columns=list(df_name))
    df_name.to_csv(performance_file, mode='a+', index=None)

    ############## Training and Testing ################
    
    best_corr = 0

    for epoch in range(1, epoch_num):

        # decay
        if epoch % decay_step == 0:
            for p in optimizer.param_groups:
                p['lr'] *= decay_rate


        print(f"Epoch: {epoch}")
        train(model, optimizer, train_loader, train_score_input, device)
        train_corr, train_rmse, train_mae, train_output, train_true  = test(model, train_loader, train_score_input, device)
        test_corr, test_rmse, test_mae, test_output, test_true  = test(model, test_loader, test_score_input, device)
        
        
        print(f'Train_corr: {train_corr:.4f}, Train_rmse: {train_rmse:.4f}, Train_mae: {train_mae:.4f}')
        print(f'Test_corr: {test_corr:.4f},  Test_rmse: {test_rmse:.4f}, Test_mae: {test_mae:.4f}')
        
        # convert to tensor
        train_corr = torch.tensor(np.float32(train_corr))
        test_corr = torch.tensor(np.float32(test_corr))
        test_rmse = torch.tensor(np.float32(test_rmse.cpu().detach().numpy()))
        test_mae = torch.tensor(np.float32(test_mae.cpu().detach().numpy()))
        
        # write performance to df
        df =[epoch,train_rmse, train_mae, train_corr, test_rmse, test_mae,test_corr]
        df = torch.Tensor(df)
        df = pd.DataFrame(np.reshape(df.cpu().detach().numpy(),(1,7)))
        df.to_csv(performance_file, mode='a+',header=None,index=None) 

        if best_corr < test_corr:
            best_train_output = train_output
            best_train_true = train_true
            best_corr = test_corr
            best_rmse = test_rmse
            best_mae = test_mae
            best_test_output = test_output
            best_test_true = test_true
            ###############save best model parameters ############### 
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     }, PATH)
    

    # append metrics
    test_corrs.append(best_corr)
    test_rmses.append(best_rmse)
    test_maes.append(best_mae)

    # save train performance to csv     
    df_predict = {'Predicted': best_train_output, 'Actual': best_train_true}
    df_predict = pd.DataFrame(data=df_predict, dtype=np.float32)
    df_predict.to_csv(predict_train_csv, mode='a+', header=True) 

    # save test performance to csv
    df_predict2 = {'Predicted':best_test_output, 'Actual':best_test_true}
    df_predict2 = pd.DataFrame(data=df_predict2, dtype=np.float32)
    df_predict2.to_csv(predict_test_csv, mode='a+', header=True)

    # append to overall predictions and actual values 
    pred_out = np.concatenate((pred_out, best_test_output), axis=0)
    true_out = np.concatenate((true_out, best_test_true), axis=0)


###############whole prediction performance###########################
corr = torch.tensor(np.mean(np.array(test_corrs,dtype=np.float32)))
rmse = torch.tensor(np.mean(np.array(test_rmses,dtype=np.float32)))
mae = torch.tensor(np.mean(np.array(test_maes,dtype=np.float32)))

# save all predictions over folds as one file
df_predict3 = {'Predicted':torch.tensor(pred_out), 'Actual':torch.tensor(true_out)}
df_predict3 = pd.DataFrame(data=df_predict3, dtype=np.float32)
df_predict3.to_csv(prediction_all_folds_csv, mode='a+', header=True)

# save final metrics
final = [corr, rmse, mae]
print(corr, rmse, mae)

final = torch.Tensor(final)
final = pd.DataFrame(final)
final.to_csv(finalfile, mode='a+', header=True) 

#######################clear cache###########################
torch.cuda.empty_cache()
torch.cuda.empty_cache()

