import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from sklearn import preprocessing
import heapq



def create_dataset_FC(data, indexx):
    """
    This function processes FC data by normalizing it, extracting features,
    selecting the top 10% of connections based on their strength, and creating
    a graph data structure for each subject

    Parameters:
    - data: 3D numpy array of size 839x116x116 containing FC data for each 839 participants
    - indexx: The indices corresponding to each subject

    Returns:
    - list: A list of Data objects (from PyTorch Geometric) where each Data
      object represents a graph constructed from the FC data of a subject
    """
    dataset_list = []
    kk = 11#22 17
    for i in range(len(data)):
        feature_matrix_ori = np.array(data[i])
        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
        feature_matrix = feature_matrix_ori2[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
       
        # gettop 10%
        edge_index_coo = np.triu_indices(116, k=1)
        edge_adj = np.zeros((116, 116))
        for ii in range(len(feature_matrix_ori2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
       
        edge_weight = edge_adj[edge_index_coo]
        edge_index_coo_array = np.array(edge_index_coo)
        edge_index_coo = torch.tensor(edge_index_coo_array)

       
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, edge_weight=torch.tensor(edge_weight, dtype = torch.float32), y = torch.tensor(indexx[i]))
        dataset_list.append(graph_data)
    return dataset_list

def create_dataset_SC(data, indexx):
    """
    This function processes SC data by normalizing it, extracting features, and creating
    a graph data structure for each subject.
    Edges in the graph are determined by the strength of connections, with a threshold (0.6) applied.

    Parameters:
    - data: 3D numpy array of size 839x116x116 containing SC data for each 839 participants
    - indexx: The indices corresponding to each subject

    Returns:
    - list: A list of Data objects (from PyTorch Geometric) where each Data
      object represents a graph constructed from the FC data of a subject
    """
    dataset_list = []
    n = data.shape[1]
    
    for i in range(len(data)):
        feature_matrix_ori = np.array(data[i])
        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
        feature_matrix = feature_matrix_ori2[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        
        edge_index_coo = np.triu_indices(116, k=1)
        edge_weight = feature_matrix_ori2[edge_index_coo] 
        
        # get top 0.6 for SC
        edge_weight[edge_weight<0.6] = 0   ##for SC  #7
        edge_index_coo = torch.tensor(edge_index_coo)
        
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, edge_weight=torch.tensor(edge_weight, dtype = torch.float32), y = torch.tensor(indexx[i]))
        dataset_list.append(graph_data)
    return dataset_list

def create_fusion_end_dataset(data_FC, data_SC, indexx):

    """
    This function processes both FC and SC data by normalizing and extracting
    features from them. It then fuses these features and constructs a graph
    data structure for each subject, considering the top 10% of connections
    for FC data and strength threshold of 0.6 for SC data.

    Parameters:
    - data_FC: 3D numpy array of size 839x116x116 containing FC data for each 839 participants
    - data_SC: 3D numpy array of size 839x116x116 containing SC data for each 839 participants
    - indexx: The indices corresponding to each subject

    Returns:
    - list: A list of Data objects (from PyTorch Geometric) where each Data
      object represents a fused graph constructed from both FC and SC data
      of a subject/sample.
    """

    dataset_list = []
    n = data_FC.shape[1]
    kk = 11#22 17 #preserve top 10% edges
    for i in range(len(data_FC)):
        # arrays
        feature_matrix_ori_FC = np.array(data_FC[i])
        feature_matrix_ori_SC = np.array(data_SC[i])

        # feature matrix for FC
        feature_matrix_ori_FC2 =  feature_matrix_ori_FC/np.max(feature_matrix_ori_FC)
        feature_matrix_FC = feature_matrix_ori_FC2[~np.eye(feature_matrix_ori_FC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_FC2.shape[0],-1)

        # feature matrix for SC
        feature_matrix_ori_SC2 =  feature_matrix_ori_SC/np.max(feature_matrix_ori_SC)
        feature_matrix_SC = feature_matrix_ori_SC2[~np.eye(feature_matrix_ori_SC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_SC2.shape[0],-1)
        
        # concatenate FC and SC feature matrices
        feature_matrix_total = np.concatenate([feature_matrix_FC, feature_matrix_SC], axis = 0)

        # edges for FC and SC

        # get upper triangular indices
        edge_index_FC = np.triu_indices(116, k=1)
        edge_index_SC = np.triu_indices(116, k=1)

        # FC edges
        # get 116x116 zeros 
        edge_adj = np.zeros((116, 116))
        for ii in range(len(feature_matrix_ori_FC2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori_FC[ii])),feature_matrix_ori_FC[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori_FC2[ii][index_max]
        edge_weight_FC = edge_adj[edge_index_FC]
        
        # SC edges
        edge_weight_SC = feature_matrix_ori_SC2[edge_index_SC]
        edge_weight_SC[edge_weight_SC<0.6] = 0
        edge_weight_ones = np.ones(116,int)
        edge_weight_total = np.concatenate([edge_weight_FC, edge_weight_SC, edge_weight_ones], axis = 0)

        # indexes
        edge_index_coo_array = np.array(edge_index_FC)
        edge_index_FC = torch.tensor(edge_index_coo_array)

        edge_index_coo2_array = np.array(edge_index_SC)
        edge_index_SC = torch.tensor(edge_index_coo2_array)

        edge_index_SC = edge_index_SC+116
        edge_index_coo4=np.reshape(np.arange(0,116), (1,116))
        edge_index_coo5=np.reshape(np.arange(116,232), (1,116))
        edge_index_coo3 = torch.tensor(np.concatenate((edge_index_coo4,edge_index_coo5),axis=0))        
        edge_index_coo_total = torch.cat((edge_index_FC, edge_index_SC, edge_index_coo3), 1)
         
        graph_data = Data(x = torch.tensor(feature_matrix_total, dtype = torch.float32), edge_index=edge_index_coo_total, edge_weight=torch.tensor(edge_weight_total, dtype = torch.float32),  y = torch.tensor(indexx[i])) 
        dataset_list.append(graph_data)
    return dataset_list

def create_fusion_end_dataset2(data_FC, data_SC, indexx):
    """
    This function processes both FC and SC data by normalizing and extracting
    features from them. It then fuses these features and constructs a graph
    data structure for each subject, considering the top 10% of connections
    for FC and SC data.

    ONLY DIFFERENCE TO create_fusion_end_dataset: for SC, top 10% connections are preserved.
    
    Parameters:
    - data_FC: 3D numpy array of size 839x116x116 containing FC data for each 839 participants
    - data_SC: 3D numpy array of size 839x116x116 containing SC data for each 839 participants
    - indexx: The indices corresponding to each subject

    Returns:
    - list: A list of Data objects (from PyTorch Geometric) where each Data
      object represents a fused graph constructed from both FC and SC data
      of a subject/sample.
    """
    dataset_list = []
    n = data_FC.shape[1]
    kk = 11#22 17 #preserve top 10% edges
    for i in range(len(data_FC)):
        # arrays
        feature_matrix_ori_FC = np.array(data_FC[i])
        feature_matrix_ori_SC = np.array(data_SC[i])

        # feature matrix for FC and SC
        feature_matrix_ori_FC2 =  feature_matrix_ori_FC/np.max(feature_matrix_ori_FC)
        feature_matrix_ori_SC2 =  feature_matrix_ori_SC/np.max(feature_matrix_ori_SC)
        feature_matrix_FC = feature_matrix_ori_FC2[~np.eye(feature_matrix_ori_FC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_FC2.shape[0],-1)
        feature_matrix_SC = feature_matrix_ori_SC2[~np.eye(feature_matrix_ori_SC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_SC2.shape[0],-1)
        
        # concatenate FC and SC feature matrices
        #print(np.max(node_SC),np.min(node_SC))
        feature_matrix_total = np.concatenate([feature_matrix_FC, feature_matrix_SC], axis = 0)
        #print(feature_matrix_total.shape)

        # edges for FC and SC
        # get upper triangular indices
        edge_index_FC = np.triu_indices(116, k=1)
        edge_index_SC = np.triu_indices(116, k=1)
        
        edge_adj = np.zeros((116, 116))
        edge_adj2 = np.zeros((116, 116))


        for ii in range(len(feature_matrix_ori_FC2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori_FC[ii])),feature_matrix_ori_FC[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori_FC2[ii][index_max]
        edge_weight_FC = edge_adj[edge_index_FC]

        # # values greater than 0.6 are considered as edges
        # edge_weight_SC = feature_matrix_ori_SC2[edge_index_SC]
        # edge_weight_SC[edge_weight_SC<0.6] = 0

        for ii in range(len(feature_matrix_ori_SC2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori_SC[ii])),feature_matrix_ori_SC[ii].take)
            edge_adj2[ii][index_max] = feature_matrix_ori_SC2[ii][index_max]
        edge_weight_SC = edge_adj2[edge_index_SC]

        edge_weight_ones = np.ones(116,int)
        edge_weight_total = np.concatenate([edge_weight_FC, edge_weight_SC, edge_weight_ones], axis = 0)

        # indexes
        edge_index_coo_array = np.array(edge_index_FC)
        edge_index_FC = torch.tensor(edge_index_coo_array)

        edge_index_coo2_array = np.array(edge_index_SC)
        edge_index_SC = torch.tensor(edge_index_coo2_array)

        edge_index_SC = edge_index_SC+116
        edge_index_coo4=np.reshape(np.arange(0,116), (1,116))
        edge_index_coo5=np.reshape(np.arange(116,232), (1,116))
        edge_index_coo3 = torch.tensor(np.concatenate((edge_index_coo4,edge_index_coo5),axis=0))        
        edge_index_coo_total = torch.cat((edge_index_FC, edge_index_SC, edge_index_coo3), 1)
         
        graph_data = Data(x = torch.tensor(feature_matrix_total, dtype = torch.float32), edge_index=edge_index_coo_total, edge_weight=torch.tensor(edge_weight_total, dtype = torch.float32),  y = torch.tensor(indexx[i])) 
        dataset_list.append(graph_data)
    return dataset_list

