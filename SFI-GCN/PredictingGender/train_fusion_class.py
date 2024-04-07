import torch
import numpy as np
import os
import pandas as pd
import time
import csv
from sklearn.model_selection import StratifiedKFold

from utils import create_fusion_end_dataset2
from torch_geometric.data import DataLoader
from model import SC_FC_Inter_GCN_Class
import torch.nn
import torch.nn.functional as F

#################### Functions ####################

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def write_attention(filename, train_attention, fold):
    filename1 = filename + '/train_attention' + str(fold) + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + ".npy"
    np.save(filename2, mean_train) 
  
def write_attention_test(filename, test_all):
    filename1 = filename + '/test_attention.npy'
    np.save(filename1, test_all)
    mean_train = np.mean(test_all, axis=0)
    filename2 = filename + '/test_mean_attention.npy' 
    np.save(filename2, mean_train)

def log_trans(x):
    zero_mask = x == 0
    x[~zero_mask] = np.log(x[~zero_mask])
    x[zero_mask] = 0
    return x

################### Load Data ####################
print("-----------------Loading Data---------------------")

fc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
sc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'

gender = '/home/ananya012/MainCodebase/SFI-GCN/scores/Gender_HCP.npy'

fc = np.load(fc)
sc = np.load(sc)
gender = np.load(gender)

print("-----------------Data Loaded-------------------")

################## Parameter Setting ##################

# for model
hidden_channels = 32
# based on gcn_hidden_channels, the neurons are flattened so hc3 = hidden_channels*116*2
hc3 = hidden_channels*116*2
hc2 = 128
hc4 = 230 # 115*2
bottleneck = 29

# training
epoch_num = 50
decay_rate = 0.5
decay_step = 6
lr = 0.0008
num_folds = 5
batch_size = 5

#################### File Path ####################
data_dir = '/data/ananya012/results/Gender/SFI-GCN-Gender-'+str(int(time.time())) ##need to modify
os.mkdir(data_dir)


# text file to outline parameteres used in batch of runs
param_file = data_dir + '/parameters.txt'
with open(param_file, 'w') as f:
    f.write(f"Epochs: {epoch_num}\n")
    f.write(f"Decay Rate: {decay_rate}\n")
    f.write(f"Decay Step: {decay_step}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Number of Folds: {num_folds}\n")
    f.write(f"Hidden Channels: {hidden_channels}\n")
    f.write(f"hc2: {hc2}\n")
    f.write(f"hc3: {hc3}\n")
    f.write(f"hc4: {hc4}\n")
    f.write(f"Bottleneck: {bottleneck}\n")
    f.write(f"Number of folds: {num_folds}\n")


#################### Train and Test Functions ####################
def train(model, optimizer, train_loader, train_label, device):
    model.train()
    for data in train_loader:
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        output, _ = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)

        # data.y are the indexes part of the train set. get the actual values
        labels = train_label[data.y]
        # convert to longtensor
        labels = labels.long()

        loss = F.cross_entropy(output, labels)
        regularized_loss = loss + L2Loss(model, 0.001) 

        if loss == 'nan':
            break

        regularized_loss.backward() # gradients based on loss, backpropagation
        optimizer.step() # update weights
        optimizer.zero_grad() # clear gradients

def test(model, test_loader, test_label, device):
    model.eval()
    predictions = []
    targets = []
    overall_attention = []
    total_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device)
            output, att_weight = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)

            # data.y are the indexes part of the test set. get the actual values
            labels = test_label[data.y]
            # convert to longtensor
            labels = labels.long()

            # calculate loss
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item() 

            output = torch.argmax(output, dim = 1)
            predictions.append(output)
            targets.append(labels)
            overall_attention.append(att_weight)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    overall_attention = torch.cat(overall_attention, dim=0)

    # bring back to cpu and numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    overall_attention = overall_attention.cpu().numpy()

    loss = total_loss / len(test_loader)
    accuracy = np.mean(predictions == targets)
    
    return loss, accuracy, predictions, targets, overall_attention

#################### Main Code ####################
final_accuracies = data_dir + '/final_result.csv'              
fold_accuracies = data_dir + '/fold_accuracies.txt'

stratifiedKFolds = StratifiedKFold(n_splits = num_folds, shuffle = True)
print("\n--------Split and Data loaded-----------\n")

fold = 0
accuracies = []
wait = 0
for X_train, X_test in stratifiedKFolds.split(fc, gender): 

    ############## Model Definition ################
    fold = fold+1
    model = SC_FC_Inter_GCN_Class(hidden_channels, hc2, hc3, hc4, bottleneck)
    print("Model:\n\t",model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    model.to(device)

    print(f"Fold: {fold}")

    train_data_fc = fc[X_train]
    test_data_fc = fc[X_test]
    train_data_sc = sc[X_train]
    test_data_sc = sc[X_test]

    train_labels = gender[X_train]
    test_labels = gender[X_test]

    # creates indices up to length of train and test set, then reshapes it to (len, 1)
    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))

    train_dataset = create_fusion_end_dataset2(train_data_fc, train_data_sc, index_train)
    test_dataset = create_fusion_end_dataset2(test_data_fc, test_data_sc, index_test)

    # convert to tensor and push labels to device
    train_label_tensor = torch.tensor(train_labels).to(device)
    test_label_tensor = torch.tensor(test_labels).to(device)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ############## Fold Specific File Definitions ################    
    
    
    predict_test_csv = data_dir + f'/predict_test{fold}.csv'
    performance_file = data_dir+'/performance_test'+str(fold)+'.csv' 
    df_name = {'epoch','train_loss','train_accu','test_loss','test_accu'}
    df_name = pd.DataFrame(columns=list(df_name))
    df_name.to_csv(performance_file, mode='a+', index=None)
    
    ############## Training and Testing ################
    wait = 0
    prev_best_loss = 1
    cur_best_accuracy = 0
    cur_best_loss = 1000000
    for epoch in range(1, epoch_num+1):

        # decay 
        if epoch % decay_step == 0:
            for p in optimizer.param_groups:
                p['lr'] *= decay_rate
        
        print(f"Epoch: {epoch}")
        train(model, optimizer, train_loader, train_label_tensor, device)
        train_loss, train_accuracy, train_predictions, train_targets, train_attention = test(model, train_loader, train_label_tensor, device)
        test_loss, test_accuracy, test_predictions, test_targets, test_attention = test(model, test_loader, test_label_tensor, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # write performance to df
        df = [epoch, train_loss, train_accuracy, test_loss, test_accuracy]
        df = torch.Tensor(df)
        df = pd.DataFrame(np.reshape(df.cpu().detach().numpy(),(1,5)))
        df.to_csv(performance_file, mode='a+',header=None,index=None) 
        if test_accuracy > cur_best_accuracy:
            print("!!!")
            cur_best_accuracy = test_accuracy
            best_predictions = test_predictions
            best_targets = test_targets
            best_attention = train_attention
            # torch.save(model.state_dict(), new_data_dir + '/best_model.pth')
        
        if test_loss < cur_best_loss:
            cur_best_loss = test_loss
            wait = 0
        else:
            wait += 1
            if wait == 10:
                print("Early Stopping occured at", epoch)
                break
        
    accuracies.append(test_accuracy)
    
    # epochs done

    ############### save the interactive weights for visualization (train)##########      
    write_attention(data_dir, best_attention, fold)

    ###############save prediction performance for each fold#################
    print("Saving Best Test Predictions")
    with open(predict_test_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Predictions', 'Targets'])
        for pred, tar in zip(best_predictions, best_targets):
            writer.writerow([pred, tar])

# write accuracies to final file
overall_accuracy = np.mean(accuracies)
with open(final_accuracies, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Overall Accuracy', overall_accuracy])

# write fold accuracies
with open(fold_accuracies, 'w') as f:
    for i, acc in enumerate(accuracies):
        f.write(f"Fold {i+1}: {acc}\n")


#######################clear cache###########################
torch.cuda.empty_cache()

