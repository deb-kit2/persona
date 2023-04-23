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
        

    def forward(self, x, persona, adj_hat, last, conv_lens) :
        # takes in conversations, and adjacency matrices in batched format.
        # x : conversation : batch, n_nodes, d_in
        # persona : batch, 4, d_in 
        # adj_hat : batch, 4 + n_nodes, 4 + n_nodes

        attented, _ = self.mha(x, x, x, need_weights = False)
        # add positional embedding?
        
        x = torch.cat((attented, persona), dim = -2)

        x = f.dropout(attented, self.dropout)
        x = self.gcn1(x, adj_hat)

        x = f.dropout(x, self.dropout)
        x = self.gcn2(x, adj_hat)

        cls = torch.index_select(x, dim = 1, index = conv_lens) # test


        return
