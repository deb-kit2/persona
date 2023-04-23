import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.graphs import GCNLayer, GCNLayerOrig, GATLayer
from modules.decoders import BARTDecoder, T5Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class PersonaModel(nn.Module) :
    def __init__(self, args) :
        self.__init__()

        self.dropout = args.dropout

        if args.pretrained_name.endswith("base") :
            self.d_in = 768
        elif args.pretrained_name.endswith("large") :
            self.d_in = 1024
        else :
            self.d_in = 512

        self.mha = nn.MultiheadAttention(batch_first = True, embed_dim = self.d_in, 
                                         num_heads = args.heads, dropout = self.dropout,
                                         device = device)
        
        self.gcn1 = GCNLayerOrig(self.d_in, self.d_in)
        self.gcn2 = GCNLayerOrig(self.d_in, self.d_in)
        
        self.decoder = BARTDecoder.from_pretrained(args.pretrained_name)
        

    def forward(self, x, adj_hat) :
        # takes in conversations, and adjacency matrices in batched format.
        # n_nodes = n_turns in a conversation
        # x : conversations : batch, n_nodes
        # adj_hat : adj matrices : batch, n_turns, n_nodes

        x = f.dropout(x, self.dropout)
        x = f.dropout(x, self.dropout)

        return
