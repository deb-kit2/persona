import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.graphs import GCNLayer, GCNLayerOrig, GATLayer
from modules.decoders import BARTDecoder, T5Decoder


class PersonaModel(nn.Module) :
    def __init__(self, args) :
        self.__init__()

        self.batch_size = args.batch_size
        self.num_heads = args.num_heads
        self.max_conv_length = args.max_conv_length

        self.dropout = args.dropout

        if args.pretrained_name.endswith("base") :
            self.d_in = 768
        elif args.pretrained_name.endswith("large") :
            self.d_in = 1024
        else :
            self.d_in = 512

        self.mha = nn.MultiheadAttention(batch_first = True, embed_dim = self.d_in, 
                                         num_heads = args.heads, dropout = self.dropout
                                        )
        
        self.gcn1 = GCNLayerOrig(self.d_in, self.d_in)
        self.gcn2 = GCNLayerOrig(self.d_in, self.d_in)
        
        self.decoder = BARTDecoder.from_pretrained(args.pretrained_name)
        self.lm_head = nn.Linear(self.d_in, self.decoder.shared.num_embeddings, bias = False)
        

    def forward(self, x, persona, adj_hat, mask,
                encoder_hidden_states = None, encoder_attention_mask = None,
                decoder_input_ids = None, decoder_attention_mask = None) :
        # takes in conversations, and adjacency matrices in batched format.
        # x : conversation : batch, 50, d_in
        # persona : batch, 6, d_in 
        # adj_hat : batch, 6 + 50, 6 + 50

        mask_ = mask.unsqueeze(dim = 1).repeat(self.num_heads, self.max_conv_length, 1)
        attented, _ = self.mha(x, x, x, 
                               need_weights = False, 
                               attn_mask = mask_)
        # add positional embedding?

        x = attented * (~ mask.unsqueeze(dim = 2))
        x = torch.cat((persona, attented), dim = -2)

        x = f.dropout(attented, self.dropout)
        x = self.gcn1(x, adj_hat)
        x = f.dropout(x, self.dropout)
        x = self.gcn2(x, adj_hat)

        cls = x[:, -1, :]
        encoder_hidden_states[:, 0, :] = cls
        x = self.decoder(
            input_ids = decoder_input_ids, 
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask
        )
        x = x[0]

        x = self.lm_head(x)

        return x
