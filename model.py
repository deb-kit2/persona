import logging
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

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, mask = None) :
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = f.softmax(wei, dim = -1)

        out = wei @ v
        return out


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
        
        self.conv_mha = nn.MultiheadAttention(batch_first = True, embed_dim = self.d_in, 
                                              num_heads = args.num_heads, dropout = self.dropout
                                              )
        self.gcn1 = GCNLayerOrig(self.d_in, self.d_in)
        self.gcn2 = GCNLayerOrig(self.d_in, self.d_in)
        
        self.h_dense = nn.Linear(self.d_in, 4 * self.d_in)
        self.h_dense2 = nn.Linear(4 * self.d_in, self.d_in)

        self.r_enc_mha = nn.MultiheadAttention(batch_first = True, embed_dim = self.d_in,
                                               num_heads = args.num_heads, dropout = self.dropout
                                               )

        self.head = Head(self.d_in, args.max_length)

        self.o_dense = nn.Linear(self.d_in, 4 * self.d_in)
        self.o_dense2 = nn.Linear(4 * self.d_in, self.d_in)
        
        self.lm_head = nn.Linear(self.d_in, args.vocab_size, bias = False)

    def forward(self,
                conv_cls, 
                persona_cls,
                adj,
                conv_mask,
                labels: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                ) :
        
        # add positional embedding?
        mask_ = conv_mask.unsqueeze(dim = 1).repeat(self.num_heads, self.max_conv_length, 1)
        attented, _ = self.conv_mha(conv_cls, conv_cls, conv_cls, 
                                    need_weights = False, 
                                    attn_mask = mask_)
        
        x = attented * (~ conv_mask.unsqueeze(dim = 2))
        x = torch.cat((persona_cls, x), dim = -2)

        x = f.dropout(x, self.dropout)
        x = self.gcn1(x, adj)
        x = f.dropout(x, self.dropout)
        x = self.gcn2(x, adj) 
        x = f.dropout(x, self.dropout)

        # before feed forward
        h_enc = self.h_dense(x) # batch, n_nodes, 4d
        h_enc = self.h_dense2(f.leaky_relu(h_enc, 0.2)) + x # batch, n_nodes, d
        # To-Do normalization

        h_enc_mask = None # To-Do

        if labels is None :
            return {
                "h_enc" : h_enc,
            }
    
        r = self.embedding_table(decoder_input_ids)
        # add positional embedding?

        # To-Do use decoder_attention_mask
        h_r = self.head(r, decoder_attention_mask)
        
        o = self.r_enc_mha(h_r, h_enc, h_enc) # b, t, d
        o_ = self.o_dense(o) # b, t, 4d
        o_ = self.o_dense2(f.leaky_relu(o_, 0.2)) + o # b, t, d

        # To-Do normalization

        logits = self.lm_head(o_) # batch, t, vocab_size
        return {
            "h_enc" : h_enc,
            "logits" :  logits
        }
    
    def generate(self, inputs = None, max_new_tokens = 32) :
        # inputs : batch, t
        for _ in range(max_new_tokens) :

        return p
