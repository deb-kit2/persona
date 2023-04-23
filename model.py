import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f

# T5
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG

# bart
from transformers import BARTConfig, BARTForConditionalGeneration

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)

from modules.graphs import paperGCN, vanillaGCN, multiHeadGAT


device = "cuda" if torch.cuda.is_available() else "cpu"


class PersonaModelT5(T5ForConditionalGeneration) :
     
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config : T5Config) :
        super().__init__(config)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(self, input_ids)

class PersonaModel(nn.Module) :
    def __init__(self, args) :
        self.__init__()

        # self.encoder = 
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
