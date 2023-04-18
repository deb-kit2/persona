import json

import torch
from torch.utils.data import Dataset

def construct_adj(max_n, n_nodes, n_personas = 4) :
    # returns the unweighted and relation-unaware graph
    
    adj = torch.zeros((max_n, max_n))

    return adj


class PersonaDataset(Dataset) :
    # Dataset class for the PersonaChat dataset

    def __init__(self, args, tokenizer, subset = "train") :
        
        self.tokenizer = tokenizer
        self.max_len = args.max_length

        DATA_PATH = ""
        with open(DATA_PATH, "r", encoding = "utf-8") as fi :
            data = json.load(fi)[subset]
        
        self.conv_id = []
        self.conv_lens = []
        self.conversation = []
        self.target = []
        self.persona = []

    def __len__(self) :
        return len(self.conv_id)
    
    def __getitem__(self, index) :

        conversation = self.tokenizer.batch_encode_plus(
            self.conversation[index],
            max_length = self.max_len,
            pad_to_max = True,
            truncation = True, 
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )

        target = self.tokenizer.batch_encode_plus(
            [self.target[index]],
            max_length = self.max_len,
            pad_to_max = True,
            truncation = True, 
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = True,
            verbose = False
        )

        persona = self.tokenizer.batch_encode_plus(
            self.persona[index],
            max_length = self.max_len,
            pad_to_max = True,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = False,
            verbose = False
        )

        return {
            "conv_id" : self.conv_id[index],
            "conversation_input_ids" : conversation["input_ids"],
            "conversation_attention_mask" : conversation["attention_mask"],
 
            "target_input_ids" : target["input_ids"].squeeze(),
            "target_attention_mask" : target["attention_mask"].squeeze(),

            "persona_input_ids" : persona["input_ids"],
            "person_attention_mask" : persona["attention_mask"],

            "adj" : ,
        }