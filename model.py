import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.graphs import GCNLayer, GCNLayerOrig, GATLayer


class Head(nn.Module) :
    def __init__(self, dim, block_size) :
        super().__init__()

        self.key = nn.Linear(dim, dim, bias = False)
        self.query = nn.Linear(dim, dim, bias = False)
        self.value = nn.Linear(dim, dim, bias = False)

        self.dropout = nn.Dropout(0.1)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, q, k, v, mask = None, causal = True) :
        B, T, C = q.shape

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, T
        if causal :
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        if mask is not None :
            wei = wei.masked_fill(mask.unsqueeze(dim = 1).repeat(1, T, 1) == 1, float("-inf"))
        wei = f.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MHA(nn.Module) :
    def __init__(self, n_heads, d) :
        super().__init__()
        
        self.heads = nn.ModuleList([Head(d, 32) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * d, d)
        self.droput = nn.Dropout(0.1)

    def forward(self, q, k, v, mask = None, causal = True) :
        out = torch.cat([h(q, k, v, mask, causal) for h in self.heads], dim = -1)
        out = self.linear(out)
        out = self.droput(out)
        
        return out


class AddPositionalEncoding(nn.Module) :

    def __init__(self, d, max_len = 128) :
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, max_len, d)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)
    

class FeedForward(nn.Module) :
    def __init__(self, d) :
        super().__init__()

        self.dense1 = nn.Linear(d, 4 * d)
        self.dense2 = nn.Linear(4 * d, d)

        self.ln = nn.LayerNorm(d)

    def forward(self, x) :
        y = self.dense1(x)
        y = f.relu(y)
        y = self.dense2(y) + x
        y = self.ln(y)

        return y


class PersonaAwareGeneration(nn.Module) :
    def __init__(self, args) :
        super().__init__()

        self.num_heads = args.num_heads
        self.max_conv_length = args.max_conv_length
        self.dropout = args.dropout

        if args.pretrained_name.endswith("base") :
            self.d_in = 768
        elif args.pretrained_name.endswith("large") :
            self.d_in = 1024
        else : # small
            self.d_in = 512
        
        self.conv_mha = MHA(args.num_heads, self.d_in)

        self.gcn1 = GCNLayerOrig(self.d_in, self.d_in)
        self.gcn2 = GCNLayerOrig(self.d_in, self.d_in)
        
        self.h_dense = nn.Linear(self.d_in, 4 * self.d_in)
        self.h_dense2 = nn.Linear(4 * self.d_in, self.d_in)

        self.r_mha = MHA(args.num_heads, self.d_in)
        self.r_enc_mha = MHA(args.num_heads, self.d_in)

        self.graph_ff = FeedForward(self.d_in)
        self.logits_ff = FeedForward(self.d_in)
        
        self.lm_head = nn.Linear(self.d_in, args.vocab_size, bias = False)

        self.pos_encoding = AddPositionalEncoding(self.d_in)
        
        self.embedding_table = nn.Embedding(args.vocab_size, self.d_in, padding_idx = 1)
        self.embedding_table.weight = torch.load(args.embedding_path)
        self.embedding_table.weight.requires_grad = False
        

    def forward(self,
                conv_cls, # B, 50, d
                persona_cls, # B, 6, d
                adj, # B, 56, 56
                conv_mask, # B, 50
                h_enc_mask, # B, 56
                labels: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None, # B, 32
                decoder_attention_mask: Optional[torch.Tensor] = None, # B, 32
                ) :
        
        conv_cls = self.pos_encoding(conv_cls)
        attented = self.conv_mha(conv_cls, conv_cls, conv_cls,
                                 mask = conv_mask, causal = False)
        
        x = attented * (~ conv_mask.unsqueeze(dim = 2)) # B, 50, d
        x = torch.cat((persona_cls, x), dim = -2) # B, 56, d

        x = f.dropout(x, self.dropout)
        x = self.gcn1(x, adj)
        x = f.dropout(x, self.dropout)
        x = self.gcn2(x, adj) 
        x = f.dropout(x, self.dropout)

        h_enc = self.graph_ff(x) # B, 56, d

        if labels is None :
            return {
                "h_enc" : h_enc,
                "h_enc_mask" : h_enc_mask
            }
    
        r = self.embedding_table(decoder_input_ids) # B, 32, d
        r = self.pos_encoding(r)
        h_r = self.r_mha(r, r, r, 
                         mask = decoder_attention_mask, causal = True) # B, 32, d 
        
        o = self.r_enc_mha(h_r, h_enc, h_enc,
                           mask = h_enc_mask, causal = False) # B, 32, d
        o = self.logits_ff(o)
        logits = self.lm_head(o) # B, 32, v

        return {
            "h_enc" : h_enc,
            "h_enc_mask" : h_enc_mask,
            "logits" :  logits
        }
    
    def generate(self, inputs = None, max_new_tokens = 32) :
        # inputs : batch, t
        # for _ in range(max_new_tokens) :
        # 
        return
