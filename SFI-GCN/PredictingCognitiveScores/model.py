import torch
import numpy as np
import os
from torch.nn import Linear, BatchNorm1d, Conv2d, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling
from torch_sparse import SparseTensor


class SC_FC_Inter_GCN(torch.nn.Module):
    """
    SFI-GCN Model combining functional and structural connectivity data with GCN for
    interactive learning and prediction.

    Attributes:
        conv2 (GCNConv): GCN convolution layer to process the connectivity data.
        conv1 (Conv2d): 2D convolution layer to learn interactive weights.
        lin1, lin2, lin3, lin4 (Linear): Linear layers for processing learned features.
    """
    def __init__(self, hidden_channels, hc2, hc3, hc4):
        super(SC_FC_Inter_GCN, self).__init__()

        # Seed for reproducibility
        torch.manual_seed(12345)      

        # Layers for Learning interactive weights
        self.conv1 = Conv2d(1, 1, (1, hc4))
        self.lin1 = Linear(116, 58)      
        self.lin2 = Linear(58, 116)

        # Layers for regression task
        # GCN layer
        self.conv2 = GCNConv(115, hidden_channels)
        # Layers for prediction
        self.lin3 = Linear(hc3, hc2)
        self.lin4 = Linear(hc2, 1)

    def flatten(self, x, batch):       

        """Flatten node features for each graph in the batch."""

        # Calculate segment size  
        seg = int(x.shape[0] / len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)), seg * x.shape[1])

        # Reshape x for each sample in batch 
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i * seg:(i + 1) * seg], (1, seg * x.shape[1]))

        return flatten_x
    
    def forward(self, x, edge_index, edge_weight, batch, device):      
        """
        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity in COO format.
            edge_weight: Edge weights.
            batch: Batch vector
            device: Device to which tensors will be moved.

        Returns:
            Tuple of output predictions and the learned interactive attention weights.
        """


        ###### EXTRACT NODE FEATURE ######

        # segment lengths
        seg2 = int(x.shape[0] / len(torch.unique(batch)))
        seg = int(seg2 / 2)

        # separate FC and SC features 
        node_fc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        node_sc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            node_fc[i] = torch.reshape(x[i * seg2:(i * seg2 + seg)], (1, seg, x.shape[1]))
            node_sc[i] = torch.reshape(x[(i * seg2 + seg):(i + 1) * seg2], (1, seg, x.shape[1]))
        # reshape   
        node_fc = torch.reshape(node_fc, (len(torch.unique(batch))*seg, x.shape[1]))
        node_sc = torch.reshape(node_sc, (len(torch.unique(batch))*seg, x.shape[1]))
        # send to device
        node_fc = node_fc.to(device)
        node_sc = node_sc.to(device) 
        
        # learning interactive weights
        inter_weights = torch.concat((node_fc,node_sc), 1)
        inter_weights = torch.reshape(inter_weights, ((len(torch.unique(batch))), 1, seg, inter_weights.shape[1])) 
        inter_weights = inter_weights.to(device)

        inter_weights = self.conv1(inter_weights)
        inter_weights = torch.reshape(inter_weights, ((len(torch.unique(batch))), seg))
        inter_weights = F.relu(self.lin1(inter_weights))
        inter_weights = F.relu(self.lin2(inter_weights))

        ## Update edge weights and node features with learned interactive weights
        # seg2: number of edges per graph in batch
        seg2 = int(edge_weight.shape[0]/len(torch.unique(batch)))
        edge_tem = torch.empty(len(torch.unique(batch)), seg2)            
        for i in range(0, len(torch.unique(batch))):
            edge_tem[i] = torch.reshape(edge_weight[i*seg2:(i+1)*seg2], (1, seg2))
            edge_tem[i][-116:] = inter_weights[i]          
        edge_tem = edge_tem.to(device)
        # back to original edge_weight shape
        edge_tem = torch.reshape(edge_tem, (len(edge_weight), 1))
        # 1D tensor
        edge_tem = torch.squeeze(edge_tem)
        
        ## update node features of new graph with learned node features
        # seg: number of nodes per graph in batch
        seg = int(node_fc.shape[0]/len(torch.unique(batch)))
        node_dim = node_fc.shape[1]
        # to store updated node features. size is doubled as for concatenation of FC ans SC
        x = torch.empty(len(torch.unique(batch))*seg*2, node_dim)

        # concat fc and sc node features
        for i in range(0, len(torch.unique(batch))):
            x[i*2*seg:i*2*seg+seg,:] = node_fc[seg*i:seg*(i+1),:]
            x[i*2*seg+seg:(i+1)*2*seg,:] = node_sc[seg*i:seg*(i+1),:]

        # GCN convolution
        x = x.to(device)
        x = self.conv2(x, edge_index, edge_tem)
        x = F.relu(x)

        # Flatten node features for each graph
        x = self.flatten(x, batch)
        x = x.to(device)    

        # Apply dropout and final regression layers
        x = F.dropout(x, p = 0.5, training = self.training)
        x = F.relu(self.lin3(x))
        x = self.lin4(x) 

        return x, inter_weights


class GCN(torch.nn.Module):
    """
    GCN model utilising SC or FC data to perform regression tasks on cognitive scores.

    Attributes:
        conv1 (GCNConv): GCN convolution layer to process the connectivity data.
        lin1, lin2 (Linear): Linear layers for processing learned features.
    """
    def __init__(self, hidden_channels, hc_gcn, hc2):
        super(GCN, self).__init__()

        # Seed for reproducibility
        torch.manual_seed(12345)

        self.conv1 = GCNConv(115, hidden_channels)
        self.lin1 = Linear(hc_gcn, hc2)
        self.lin2 = Linear(hc2, 1)

    def flatten(self, x, batch):      
        """Flatten node features for each graph in the batch."""
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        
        # 1. Obtain node embeddings    
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # print("Post conv1:", x.shape)  # Debug print

        # 2. Readout layer
        x = self.flatten(x, batch)
        x = x.to(device)   
        # print("Post flatten:", x.shape)  # Debug print

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        # print("Post lin1:", x.shape) 
        x = self.lin2(x) 
        return x


class MV_GCN(torch.nn.Module):
    """
    Multi-view GCN model utilising SC and FC data to perform regression tasks on cognitive scores.

    Attributes:
        conv1, conv2 (GCNConv): GCN convolution layers to process the connectivity data.
        lin6, lin7 (Linear): Linear layers for processing learned features.
    """
    def __init__(self, hidden_channels, hc):
        super(MV_GCN, self).__init__()

        # Seed for reproducibility
        torch.manual_seed(12345)      

        # GCN layers
        self.conv1 = GCNConv(115, hidden_channels)
        self.conv2 = GCNConv(115, hidden_channels)

        # Linear layers
        self.lin6 = Linear(116*2, hc)
        self.lin7 = Linear(hc, 1)

    def flatten(self, x, batch):        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):        

        ###### EXTRACT NODE FEATURE ######
        # segment lengths
        seg2 = int(x.shape[0]/len(torch.unique(batch)))
        seg = int(seg2/2)

        # separate FC and SC features 
        node_fc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        node_sc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            node_fc[i] = torch.reshape(x[i*seg2:(i*seg2+seg)], (1, seg, x.shape[1]))
            node_sc[i] = torch.reshape(x[(i*seg2+seg):(i+1)*seg2], (1, seg, x.shape[1]))        
        # reshape   
        node_fc = torch.reshape(node_fc,(len(torch.unique(batch))*seg,x.shape[1]))
        node_sc = torch.reshape(node_sc,(len(torch.unique(batch))*seg,x.shape[1]))
        node_fc = node_fc.to(device)
        node_sc = node_sc.to(device) 

        ###### EXTRACT EDGE WEIGHT ######
        # 116 * (116 - 1) // 2 = 6670
        lenn = 6670
        seg2 = int(edge_weight.shape[0]/len(torch.unique(batch)))
        seg = seg2-116
        edge_tem = torch.empty(1,seg*len(torch.unique(batch)))    
        # for fc
        edge_weight1 = torch.empty(1,lenn*len(torch.unique(batch))) 
        # for sc
        edge_weight2 = torch.empty(1,lenn*len(torch.unique(batch))) 
        for i in range(0, len(torch.unique(batch))):
            edge_tem[0,i*seg:seg*(i+1)] = torch.reshape(edge_weight[seg2*i:(seg2*i+seg)], (1, seg))
            edge_weight1[0,i*lenn:(i+1)*lenn]=edge_tem[0,i*seg:i*seg+lenn]
            edge_weight2[0,i*lenn:(i+1)*lenn]=edge_tem[0,i*seg+lenn:(i+1)*seg]
        
        edge_weight1 = torch.squeeze(edge_weight1)
        edge_weight2 = torch.squeeze(edge_weight2)
        edge_weight1 = edge_weight1.to(device)
        edge_weight2 = edge_weight2.to(device)

        ###### EXTRACT EDGE INDEX ######
        edge_tem = torch.empty(2,seg*len(torch.unique(batch)),dtype=torch.long) 
        # for fc 
        edge_index1 = torch.empty(2,lenn*len(torch.unique(batch)),dtype=torch.long) 
        # for sc
        edge_index2 = torch.empty(2,lenn*len(torch.unique(batch)),dtype=torch.long) 
        for i in range(0, len(torch.unique(batch))):    
            edge_tem[0:2,i*seg:seg*(i+1)] = edge_index[0:2,seg2*i:(seg2*i+seg)]            
            edge_index1[0:2,i*lenn:(i+1)*lenn] = edge_tem[0:2,i*seg:i*seg+lenn]-116*i
            edge_index2[0:2,i*lenn:(i+1)*lenn] = edge_tem[0:2,i*seg+lenn:(i+1)*seg]-116*(i+1)
        edge_index1 = edge_index1.to(device)
        edge_index2 = edge_index2.to(device)
        
        #Seperate Multi-view GCNs on SC and FC for prediction
        x1 = self.conv1(node_fc, edge_index1, edge_weight1)
        x2 = self.conv2(node_sc, edge_index2, edge_weight2)
        # max
        x1, _ = torch.max(x1, dim = 1)
        x2, _ = torch.max(x2, dim = 1)
        # reshape
        x1 = torch.reshape(x1, (x1.shape[0], 1))
        x2 = torch.reshape(x2, (x2.shape[0], 1))
        # concat
        x = torch.concat((x1,x2), 1)
        x = self.flatten(x, batch)
        x = x.to(device)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin6(x)
        x = F.relu(x)
        x = self.lin7(x) 
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(115, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels2)
        self.lin1 = Linear(hc, hc2)
        self.lin2 = Linear(hc2, hc2)
        self.lin3 = Linear(hc2, 1)
        self.pool1 = TopKPooling(hidden_channels2, 0.3)
        #self.double()
    def flatten(self, x, batch):
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))      
        return flatten_x
 
    def forward(self, x, edge_index, edge_weight, batch, device):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        #x = F.leaky_relu(x, negative_slope=0.33)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.leaky_relu(x, negative_slope=0.33)
        #print(x.shape)
        # 2. Readout layer
        #x = self.pool1(x, edge_index, batch = batch)  # [batch_size, hidden_channels]
        #x = x[0]
        x = self.flatten(x, batch)
        x = x.to(device)  
        #print(x[0].shape)
        #x = global_mean_pool(x,batch)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
       # print(x.shape)
        #sx.to(device)
        x = self.lin1(x)
        #x = F.leaky_relu(x, negative_slope=0.33)
        x = F.relu(x)
       # x = F.leaky_relu(x, negative_slope=0.33)
        x = self.lin3(x)
        return x
