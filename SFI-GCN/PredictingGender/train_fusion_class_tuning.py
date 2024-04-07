import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report

from utils import create_fusion_end_dataset, create_fusion_end_dataset2
from torch_geometric.loader import DataLoader
from model import SC_FC_Inter_GCN_Class
import torch.nn
from torch.utils.data.dataloader import default_collate
from torcheval.metrics import R2Score
import torch.nn.functional as F
from torcheval.metrics.aggregation.auc import AUC
from datetime import datetime

#################### Functions ####################

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def log_trans(x):
    zero_mask = x == 0
    x[~zero_mask] = np.log(x[~zero_mask])
    x[zero_mask] = 0
    return x

def write_attention(filename, train_attention, fold):
    filename1 = filename + '/train_attention' + str(fold) + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + ".npy"
    np.save(filename2, mean_train) 

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
gcn_hidden_channels = [23, 32, 46]
# based on gcn_hidden_channels, the neurons are flattened so hc3 = hidden_channels*116*2
hc2 = 128
hc4 = 230 # 115*2
bottleneck = [10, 29, 58, 77]

# training
epoch_num = 50
decay_rate = 0.5
decay_step = 6
lr = 0.0008
num_folds = 5
batch_size = 5

# which fold to load
# splitsetfold = 3

#################### File Path ####################
current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
print("Current Date and Time: ", current_datetime)

data_dir = f'/data/ananya012/results/Gender/PostFinalReport/TuningRuns/Tuning_hiddenchannels_bottleneck_' + str(current_datetime)
os.makedirs(data_dir)

# create a txt file to outline the parameters used in this batch of runs
with open(data_dir + '/parameters.txt', 'w') as f:
    f.write(f"Epochs: {epoch_num}\n")
    f.write(f"Decay Rate: {decay_rate}\n")
    f.write(f"Decay Step: {decay_step}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Number of Folds: {num_folds}\n")
    f.write(f"GCN Hidden Channels: {gcn_hidden_channels}\n")
    f.write(f"HC2: {hc2}\n")
    f.write(f"HC4: {hc4}\n")
    f.write(f"Bottleneck: {bottleneck}\n")





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



overall_performance = {}
for hidden_channels in gcn_hidden_channels:
    for bn in bottleneck:

        #################### File Definitions ####################
        cur_params = f'/hidchan_{hidden_channels}_bn_{bn}'

        new_data_dir = data_dir + cur_params
        os.makedirs(new_data_dir)

        final_accuracy = new_data_dir + '/final_accuracy.txt'
        fold_accuracies = new_data_dir + '/fold_accuracies.txt'

        accuracies = []
        #################### KFold ####################
        for fold in range(1, 6):

            ######### Model Related #########
            hc3 = hidden_channels*116*2
            model = SC_FC_Inter_GCN_Class(hidden_channels, hc2, hc3, hc4, bn)
            print("Model:\n\t", model)
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            print("Using: ", device)
            print()
            model.to(device)

            print(f"Fold: {fold}")
            print(f"Hidden Channel: {hidden_channels}, Bottleneck: {bn}")
            print()

            # load presplit
            train_idx = np.load(f'/home/ananya012/MainCodebase/SFI-GCN/PredictingGender/SplitGender/train_fold_{fold}.npy')
            test_idx = np.load(f'/home/ananya012/MainCodebase/SFI-GCN/PredictingGender/SplitGender/test_fold_{fold}.npy')

            train_data_fc = fc[train_idx]
            test_data_fc = fc[test_idx]
            train_data_sc = sc[train_idx]
            test_data_sc = sc[test_idx]

            train_labels = gender[train_idx]
            test_labels = gender[test_idx]

            # create dataset

            # creates indices up to length of train and test set, then reshapes it to (len, 1)
            index_train = np.reshape(np.arange(0, len(train_idx)), (len(train_idx), 1))
            index_test = np.reshape(np.arange(0, len(test_idx)), (len(test_idx), 1))

            # log transform SC data
            # train_data2 = log_trans(train_data2)
            # test_data2 = log_trans(test_data2)

            train_dataset = create_fusion_end_dataset2(train_data_fc, train_data_sc, train_labels, index_train)
            test_dataset = create_fusion_end_dataset2(test_data_fc, test_data_sc, test_labels, index_test)

            # conver to tensor and push labels to device
            train_label_tensor = torch.tensor(train_labels).to(device)
            test_label_tensor = torch.tensor(test_labels).to(device)

            # dataloader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # optimizer 
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            
            #################### Fold Specific Files Definitions ####################
            # predict_train_csv = new_data_dir + '/predict_train.csv'
            predict_test_csv = new_data_dir + f'/predict_test_{fold}.csv'

            #################### Training ####################
            print("-----------------Epoching---------------------")

            cur_best_accuracy = 0
            cur_best_loss = 1000000
            wait = 0
            for epoch in range(epoch_num):
                print(f"Epoch: {epoch+1}")
                train(model, optimizer, train_loader, train_label_tensor, device)
                train_loss, train_accuracy, train_predictions, train_targets, train_attention = test(model, train_loader, train_label_tensor, device)
                test_loss, test_accuracy, test_predictions, test_targets, test_attention = test(model, test_loader, test_label_tensor, device)

                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

                if epoch % decay_step == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= decay_rate
                
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
                        print("Early Stopping occured at", epoch+1)
                        break
            
            # save interactive weight for visualisation
            # write_attention(new_data_dir, best_attention, splitsetfold)
                    
            ################# save prediction performance #################
            print("Saving Best Test Predictions")
            with open(predict_test_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Predictions', 'Targets'])
                for pred, tar in zip(best_predictions, best_targets):
                    writer.writerow([pred, tar])
            
            # save classification report
            print("Classification Report")
            report = classification_report(best_targets, best_predictions, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(new_data_dir + f'/classification_report_{fold}.csv')

            print(report)

            # append best accuracy
            accuracies.append(cur_best_accuracy)   

        
        # save fold accuracies
        with open(fold_accuracies, 'w') as f:
            for i, acc in enumerate(accuracies):
                f.write(f"Fold {i+1}: {acc}\n")
        
        # save overall accuracy
        print("Saving Final Accuracy")
        mean_accuracy = round(np.mean(accuracies), 5)
        std_accuracy = round(np.std(accuracies), 5)
        with open(final_accuracy, 'w') as f:
            f.write("Mean Accuracy: " + str(mean_accuracy) + "\n")
            f.write("Std Accuracy: " + str(std_accuracy) + "\n")


        # save overall performance
        overall_performance[cur_params[1:]] = mean_accuracy



# save overall performance, sort it by accuracy
overall_performance = dict(sorted(overall_performance.items(), key=lambda item: item[1], reverse=True))
with open(data_dir + '/overall_performance.txt', 'w') as f:
    for key, value in overall_performance.items():
        f.write(f"{key}: {value}\n")

print(overall_performance)
            
            
            
            

                    
                
                




            

            
           
