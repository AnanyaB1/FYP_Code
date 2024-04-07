import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import KFold
from sklearn import preprocessing

from utils import create_dataset_FC
from torch_geometric.data import DataLoader
from model import GCN_class
import torch.nn
from torch.utils.data.dataloader import default_collate
from torcheval.metrics import R2Score
import torch.nn.functional as F
from sklearn.metrics import classification_report

#################### Functions ####################

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

################### Load Data ####################
print("-----------------Loading Data---------------------")

fc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_FC/X.npy'
sc = '/home/ananya012/MainCodebase/SFI-GCN/HCP_SC/X.npy'

gender = '/home/ananya012/MainCodebase/SFI-GCN/scores/Gender_HCP.npy'

fc = np.load(fc)
sc = np.load(sc)
gender = np.load(gender)

print("-----------------Data Loaded-------------------")

########### parameter setting ###################

hidden_channels=115 #4 #64 #115


hc_gcn = 13340
hc2 = 256 #128
hc3 = 26680 #928 #14848 #26680
hc4 = 230
epoch_num = 30
decay_rate = 0.5  
decay_step = 19
lr =0.001 
num_folds = 5
batch_size = 5
runnum = 'e1'

#################### File Path ####################

# Update base_dir to use cur_score for directory path
base_dir = f'/data/ananya012/results/GCN/Gender/FC/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


# Modified to use cur_score in the file name
timefile = f'/data/ananya012/results/GCN/Gender/FC/GCN_FC-' + str(int(time.time()))
os.mkdir(timefile)
finalfile = timefile+'/final_result.csv'         


#################### Train and Test Functions ####################
def train(model, optimizer, train_loader, train_label, device):
    model.train()
    for data in train_loader:
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        output = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)

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
    total_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device)
            output = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)

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

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    # bring back to cpu and numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    loss = total_loss / len(test_loader)
    accuracy = np.mean(predictions == targets)
    
    return loss, accuracy, predictions, targets

#################### Main Code ####################
 
num = fc.shape[0]
kf = KFold(n_splits=num_folds, shuffle=True)
print("\n--------Split and Data loaded-----------\n")
fold = 0
true_out = np.squeeze(np.array([[]]))
pred_out = np.squeeze(np.array([[]]))

accuracies = []
##############################################5-fold cross validation#########################################
for X_train, X_test in kf.split(list(range(1,num))):
    fold = fold+1
    # print("HEREEEEEEE")
    model = GCN_class(hidden_channels, hc_gcn, hc2)
    print("Model:\n\t",model)
    print(torch.cuda.is_available())
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"FC GCN on Gender")
    #device = torch.device("cpu")
    print(fold)
    model.to(device)

    
    train_data = fc[X_train]
    test_data = fc[X_test]

    train_score = gender[X_train]
    test_score = gender[X_test]

    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))
    
    training_dataset = create_dataset_FC(train_data, index_train)
    testing_dataset = create_dataset_FC(test_data, index_test)
    
    train_score_input = torch.tensor(train_score).to(device)
    test_score_input = torch.tensor(test_score).to(device)

    train_loader = DataLoader(training_dataset, batch_size, shuffle = True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(testing_dataset, batch_size, shuffle= True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr)

    ############## Fold Specific File Definitions ################    

    performance_file = timefile+'/performance_test'+str(fold)+'.csv' 
    predict_test_csv = timefile+'/predict_test'+str(fold)+'.csv'
    df_name = {'epoch','train_loss','train_accu','test_loss','test_accu'}
    df_name = pd.DataFrame(columns=list(df_name))
    df_name.to_csv(performance_file, mode='a+', index=None)


    ############## Training and Testing ################
    wait = 0
    prev_best_loss = 1
    cur_best_accuracy = 0
    cur_best_loss = 1000000
    for epoch in range(1, epoch_num+1):
        if epoch % decay_step == 0:
            for p in optimizer.param_groups:
                p['lr'] *= decay_rate

        train(model, optimizer, train_loader, train_score_input, device)
        train_loss, train_accuracy, train_predictions, train_targets  = test(model, train_loader,train_score_input, device)
        test_loss, test_accuracy, test_predictions, test_targets = test(model, test_loader,test_score_input, device)

        print(f"Epoch: {epoch}")
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
            # torch.save(model.state_dict(), new_data_dir + '/best_model.pth')
        
    accuracies.append(cur_best_accuracy)


    ###############save prediction performance for each fold#################
    print("Saving Best Test Predictions")
    with open(predict_test_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Predictions', 'Targets'])
        for pred, tar in zip(best_predictions, best_targets):
            writer.writerow([pred, tar])

###############whole prediction performance###########################
# write accuracies to final file
overall_accuracy = np.mean(accuracies)
with open(finalfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Overall Accuracy', overall_accuracy])

#######################clear cache###########################
torch.cuda.empty_cache()


