import torch
import torch.nn as nn
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"

class PersonaModel(nn.Module) :
    def __init__(self, args) :
        self.__init__()

        self.encoder = 

    def forward(self, x, adj_hat) :
        # takes in conversations, and adjacency matrices in batched format.
        # n_nodes = n_turns in a conversation
        # x : conversations : batch, n_nodes
        # adj_hat : adj matrices : batch, n_turns, n_nodes

        return decoder_outs