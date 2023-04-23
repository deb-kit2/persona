import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.graphs import GCNLayer, GCNLayerOrig, GATLayer

device = "cuda" if torch.cuda.is_available() else "cpu"


class PersonaModel(nn.Module) :
    def __init__(self, args) :
        self.__init__()

        self.mha = nn.MultiheadAttention(batch_first = True, embed_dim = 768, 
                                         num_heads = 8,dropout = 0.2,
                                         device = device)
        # self.gcn1 = 
        # self.gcn2 = 
        # self.decoder = 
        

    def forward(self, x, adj_hat) :
        # takes in conversations, and adjacency matrices in batched format.
        # n_nodes = n_turns in a conversation
        # x : conversations : batch, n_nodes
        # adj_hat : adj matrices : batch, n_turns, n_nodes

        # return decoder_outs
        return
