import numpy as np
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

y_dir = '/home/ananya012/SC-FC-fusion/gae/scores/'

scfc_dir = '/home/ananya012/HCP_Transformer/Data/'
scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]  

for score in [scores[1]]:
    x_fc = np.load(scfc_dir + "/X_FC.npy")
    x_sc = np.load(scfc_dir + "/X_SC.npy")
    y = np.load(y_dir + score + '_Unadj_HCP.npy')
    print("finished loading data")

    # from preprocess_data
    def bin_y(y, num_bins=5):
        # scale the y data
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        # bin the y data
        bins = [
            (0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0)]
        
        y_binned = np.zeros(y.shape)
        for i in range(num_bins):
            y_binned[(y >= bins[i][0]) & (y < bins[i][1])] = i

        return y_binned


    # split the data into train, test val (0.8, 0.1, 0.1)
    train_split = 0.8
    test_split = 0.2

    # split the data
    x_fc_train, x_fc_test, x_sc_train, x_sc_test, y_train, y_test = train_test_split(
        x_fc, x_sc, y, test_size=test_split, random_state=489, stratify=bin_y(y)
    )

    # x_fc_val, x_fc_test, x_sc_val, x_sc_test, y_val, y_test = train_test_split(
    #     x_fc_val, x_sc_val, y_val, test_size=test_split / (val_split + test_split), random_state=489, stratify=bin_y(y_val)
    # )

    # transform all sc 
    def log_trans(x):
        zero_mask = x == 0
        x[~zero_mask] = np.log(x[~zero_mask])
        x[zero_mask] = 0
        return x

    x_sc_train = log_trans(x_sc_train)
    x_sc_test = log_trans(x_sc_test)


    # standard scale all features
    class StandardScaler:
        def __init__(self):
            self.mean = None
            self.std = None

        def fit(self, x):
            self.mean = np.mean(x)
            self.std = np.std(x)

        def transform(self, x):
            return (x - self.mean) / self.std

        def fit_transform(self, x):
            self.fit(x)
            return self.transform(x)
        
    ss_fc = StandardScaler()
    ss_sc = StandardScaler()

    # fit both to train
    ss_fc.fit(x_fc_train)
    ss_sc.fit(x_sc_train)

    # transform all
    x_fc_train = ss_fc.transform(x_fc_train)
    x_fc_test = ss_fc.transform(x_fc_test)

    x_sc_train = ss_sc.transform(x_sc_train)
    x_sc_test = ss_sc.transform(x_sc_test)

    # set all self-loops to 0
    x_fc_train[:, np.arange(x_fc_train.shape[1]), np.arange(x_fc_train.shape[1])] = 0
    x_fc_test[:, np.arange(x_fc_test.shape[1]), np.arange(x_fc_test.shape[1])] = 0

    x_sc_train[:, np.arange(x_sc_train.shape[1]), np.arange(x_sc_train.shape[1])] = 0
    x_sc_test[:, np.arange(x_sc_test.shape[1]), np.arange(x_sc_test.shape[1])] = 0



    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)
    print(ss_y.mean, ss_y.std)

    # create a dir called score in traintestonly
    data_dir = '/home/ananya012/HCP_Transformer/Data/Train_Test_Only/' + score

    # # create dir called score
    # data_dir = '/home/ananya012/HCP_Transformer/Data/' + score

    # if data_dir doesnt exist makedir
    import os
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # store all data as processed
    np.save(data_dir + "/X_FC_TRAIN.npy", x_fc_train)
    np.save(data_dir + "/X_FC_TEST.npy", x_fc_test)

    np.save(data_dir + "/X_SC_TRAIN.npy", x_sc_train)
    np.save(data_dir + "/X_SC_TEST.npy", x_sc_test)

    np.save(data_dir + "/Y_TRAIN.npy", y_train)
    np.save(data_dir + "/Y_TEST.npy", y_test)


    # concat x_fc_train and x_sc_train
    x_train = np.concatenate((x_fc_train, x_sc_train), axis=-1)
    x_test = np.concatenate((x_fc_test, x_sc_test), axis=-1)

    class CustomDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    # create dataset
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)


    # save mean and std as npy
    np.save(data_dir + "/Y_MEAN.npy", ss_y.mean)
    np.save(data_dir + "/Y_STD.npy", ss_y.std)

    # store datasets as .pt files
    # save at data_dir
    torch.save(train_dataset, data_dir + '/train_dataset.pt')
    torch.save(test_dataset, data_dir + '/test_dataset.pt')


