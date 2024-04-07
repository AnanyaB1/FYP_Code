import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class MLPBlock(nn.Module):
    """
    Multilayer Perceptron (MLP) block with one hidden layer.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Dimensionality of the output features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        """
        Forward pass of the MLP block.

        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output of the MLP block after applying layers and ReLU activation.
        """

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    """
    Transformer module utilizing MSA mechanism.

    Args:
        model_dim (int): Dimensionality of the input and output features of the transformer.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, model_dim, num_heads):
        super(Transformer, self).__init__()
        self.mhsa = nn.MultiheadAttention(model_dim, num_heads)

    def forward(self, x):
        """
        Forward pass of the Transformer.

        Args:
            x (Tensor): Input tensor for the Multihead Self-Attention layer.

        Returns:
            Tensor: Output tensor of the Transformer module after self-attention.
        """
        x = self.mhsa(x, x, x)[0]
        return x

# block of transformer and mlp
class Block(nn.Module):
    """
    Transformer block that combines Multihead Self-Attention layer and an MLP block.

    Args:
        model_dim (int): Dimensionality of the input and output features for the transformer.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Number of neurons in the hidden layer of the MLP block.
    """
    def __init__(self, model_dim, num_heads, hidden_dim):
        super(Block, self).__init__()
        self.mlp_block = MLPBlock(model_dim, hidden_dim, model_dim)
        self.transformer = Transformer(model_dim, num_heads)

    def forward(self, x):
        """
        Forward pass of the Block.

        Args:
            x (Tensor): Input tensor for the block.

        Returns:
            Tensor: Output tensor after applying the Transformer and MLP block.
        """
        x = x + self.transformer(x)
        x = self.mlp_block(x)
        return x

class TransformerRegressor(nn.Module):
    """
    A Transformer-based model for regression tasks.

    Args:
        input_dim (int): Dimensionality of the input features.
        model_dim (int): Dimensionality of features inside the transformer.
        num_heads (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer layers (blocks) in the model.
        output_dim (int): Dimensionality of the model output.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerRegressor, self).__init__()

        self.embedding = nn.Linear(input_dim, model_dim)
        self.learned_pos_encoder = nn.Parameter(torch.zeros(1, 116, model_dim))
        self.forward_blocks = Block(model_dim, 8, 32) #nn.ModuleList([Block(model_dim, 8, 32) for _ in range(num_layers)])

        #   self.regression_head = nn.Linear(model_dim, output_dim)#
        self.regression_head = MLPBlock(model_dim, 64, output_dim)

        self.dropout = nn.Dropout(0.25)


        self.decoder = nn.Linear(model_dim, model_dim)
        self.decoder2 = nn.Linear(model_dim, input_dim)
        self.decoder3 = nn.Linear(input_dim, input_dim)


    def forward(self, x, y=None, mask=False):

        # x = x[:, :, :116] + x[:, :, 116:]

        # randomly mask 15% of input if training
        if self.training: # and mask:
            mask = torch.rand(x.size()) > 0.75
            x[mask] = 0

        # embed
        x = self.embedding(x) + self.learned_pos_encoder

        # forward blocks
        for _ in range(3): #self.forward_blocks:
            x = x + self.forward_blocks(x)

        input_recon = F.relu(self.decoder(x))
        input_recon = F.relu(self.decoder2(input_recon))
        input_recon = self.decoder3(input_recon)


        # regression head
        x = x.mean(dim=1)
        #x = x[:, 0, :]
        x = self.dropout(x)
        x = self.regression_head(x)

        return x, input_recon