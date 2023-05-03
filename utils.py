import copy
import json

import torch
from torch.utils.data import Dataset

from transformers import BartTokenizer, T5Tokenizer
from modules.encoders import BARTEncoder, T5Encoder


def construct_adj(n_nodes, n_persona, max_n = 56) :
    # returns the unweighted and relation-unaware graph
    # persona self, make bidirectional next experiment: reason persona might change over conv
    
    adj = torch.zeros((max_n, max_n))
    # to avoid div by 0 or -1 in graphs
    for i in range(max_n - n_nodes) :
        adj[i][i] = 2

    for i in range(max_n - n_nodes, max_n) :
        adj[i][i] = 1
        if i + 1 < max_n :
            adj[i + 1, i] = 1
        if i + 2 < max_n :
            adj[i + 2, i] = 1
            adj[i, i + 2] = 1
    for i in range(56 - n_nodes + 1, max_n, 2) :
        for j in range(n_persona) :
            adj[i, j] = 1

    return adj


class PersonaDataset(Dataset) :
    # Dataset class for the PersonaChat dataset

    def __init__(self, args, subset = "train") :
        
        if args.pretrained_name.endswith("base") :
            self.d_in = 768
        elif args.pretrained_name.endswith("large") :
            self.d_in = 1024
        else :
            self.d_in = 512

        self.max_len = args.max_length
        self.encrep = args.encrep
        self.max_conv_length = args.max_conv_length
        self.max_persona_length = 6

        if args.encdec == "bart" :
            # init bart
            self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_name)
            self.encoder = BARTEncoder.from_pretrained(args.pretrained_name)
        else :
            raise NotImplementedError()

        self.conv_id = []
        self.conv_lens = [] # len - 1

        self.persona = []
        self.conversation = [] # everything except target
        self.last = [] # last in conversation, <CLS> to be merged here
        self.target = [] # to be produced

        DATA_PATH = "data/personachat_data.json"
        with open(DATA_PATH, "r", encoding = "utf-8") as fi :
            data = json.load(fi)[subset]

            for sample in data :
                self.conv_id.append(sample["conv_id"])
                self.conv_lens.append(len(sample["dialog"]) - 1)

                self.persona.append(sample["personality_person2"])

                self.conversation.append([sample["dialog"][p]["text"] for p in range(len(sample["dialog"]) - 1)])
                self.last.append(sample["dialog"][-2]["text"])

                self.target.append(sample["dialog"][-1]["text"])

            del data

    def __len__(self) :
        return len(self.conv_id)
    
    def __getitem__(self, index) :

        conversation = self.tokenizer.batch_encode_plus(
            self.conversation[index],
            max_length = self.max_len,
            truncation = True, 
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )
        encoded_conv = self.encoder(conversation.input_ids, conversation.attention_mask)[0]
        if self.encrep == "first" :
            encoded_conv = encoded_conv[:, 0, :]
        else :
            encoded_conv = torch.mean(encoded_conv, dim = -2)

        dummy = torch.zeros((self.max_conv_length - self.conv_lens[index], self.d_in), dtype = torch.float32)
        encoded_conv = torch.cat((dummy, encoded_conv), dim = 0)

        mask = [True] * (self.max_conv_length - self.conv_lens[index]) + [False] * self.conv_lens[index]
        mask = torch.tensor(mask, dtype = torch.bool)

        target = self.tokenizer.batch_encode_plus(
            [self.target[index]],
            max_length = self.max_len,
            truncation = True, 
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )

        labels = copy.deepcopy(target.input_ids)
        labels = torch.tensor([[-100 if token == self.tokenizer.pad_token_id else token for token in l] for l in labels])

        persona = self.tokenizer.batch_encode_plus(
            self.persona[index],
            max_length = self.max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )
        encoded_persona = self.encoder(persona.input_ids, persona.attention_mask)[0]
        if self.encrep == "first" :
            encoded_persona = encoded_persona[:, 0, :]
        else :
            encoded_persona = torch.mean(encoded_persona, dim = -2)

        dummy = torch.zeros((self.max_persona_length - len(self.persona[index]), self.d_in), dtype = torch.float32)
        encoded_persona = torch.cat((encoded_persona, dummy), dim = 0)

        last = self.tokenizer.batch_encode_plus(
            [self.last[index]],
            max_length = self.max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )
        encoded_last = self.encoder(last.input_ids, last.attention_mask)[0]

        adj = construct_adj(self.conv_lens[index], len(self.persona[index]))

        return {
            "conv_id" : self.conv_id[index],
            "conv_cls" : encoded_conv,
            "conv_mask" : mask,
            
            "persona_cls" : encoded_persona,

            "encoder_hidden_states" : encoded_last.squeeze(),
            "encoder_attention_mask" : last.attention_mask.squeeze(),

            "decoder_input_ids" : target.input_ids.squeeze(),
            "decoder_attention_mask" : target.attention_mask.squeeze(),

            "labels" : labels.squeeze(),

            "adj" : adj
        }
    
