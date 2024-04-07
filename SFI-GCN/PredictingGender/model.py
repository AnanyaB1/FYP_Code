import torch
import numpy as np
import os
from torch.nn import Linear, BatchNorm1d, Conv2d, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import TopKPooling

class GCN_class(torch.nn.Module):
    """
    GCN Class model utilising SC or FC data to perform classification tasks on gender

    Attributes:
       conv1: GCNConv layer
       lin1, lin2: Linear layers
    """
    def __init__(self, hidden_channels, hc_gcn, hc2):
        super(GCN_class, self).__init__()
        
        # Seed for reproducibility
        torch.manual_seed(12345)

        self.conv1 = GCNConv(115, hidden_channels)
        self.lin1 = Linear(hc_gcn, hc2)
        self.lin2 = Linear(hc2, 2)

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

class SC_FC_Inter_GCN_Class(torch.nn.Module):
    """
    SFI-GCN Class Model combining functional and structural connectivity data with GCN for
    interactive learning and classification.

    Attributes:
        conv2 (GCNConv): GCN convolution layer to process the connectivity data.
        conv1 (Conv2d): 2D convolution layer to learn interactive weights.
        lin1, lin2, lin3, lin4 (Linear): Linear layers for processing learned features.
    """
    def __init__(self, hidden_channels, hc2, hc3, hc4, bottleneck):
        super(SC_FC_Inter_GCN_Class, self).__init__()

        # Seed for reproducibility
        torch.manual_seed(12345)      

        # Layers for Learning interactive weights
        self.conv1 = Conv2d(1, 1, (1, hc4))
        self.lin1 = Linear(116, bottleneck)      
        self.lin2 = Linear(bottleneck, 116)

        # Layers for regression task
        self.conv2 = GCNConv(115, hidden_channels)
        # Layers for prediction
        self.lin3 = Linear(hc3, hc2)
        self.lin4 = Linear(hc2, 2)

        # softmax layer
        self.softmax_func=Softmax(dim=1)

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

        #GCN on the joint graph
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

        # Softmax layer
        x = self.softmax_func(x) 

        return x, inter_weights  
