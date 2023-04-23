import copy
import json

import torch
from torch.utils.data import Dataset

def construct_adj(max_n, n_nodes, n_personas = 4) :
    # returns the unweighted and relation-unaware graph
    
    adj = torch.zeros((max_n, max_n))

    return adj


class PersonaDataset(Dataset) :
    # Dataset class for the PersonaChat dataset

    def __init__(self, args, subset = "train") :
        
        self.max_len = args.max_length
        self.encrep = args.encrep

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

        DATA_PATH = ""
        with open(DATA_PATH, "r", encoding = "utf-8") as fi :
            data = json.load(fi)[subset]
        
        self.dummy = torch.zeros((), dtype = torch.float32)

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
        labels = torch.tensor([[-100 if token == self.tokenier.pad_token_id else token for token in l] for l in labels])

        persona = self.tokenizer.batch_encode_plus(
            self.persona[index],
            max_length = self.max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = False,
            verbose = False
        )
        encoded_persona = self.encoder(persona.input_ids, persona.attention_mask)[0]
        if self.encrep == "first" :
            encoded_persona = encoded_persona[:, 0, :]
        else :
            encoded_persona = torch.mean(encoded_persona, dim = -2)

        last = self.tokenizer.batch_encode_plus(
            [self.last[index]],
            max_length = self.max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt",
            return_attention_mask = False,
            verbose = False
        )
        encoded_last = self.encoder(last.input_ids, last.attention_mask)[0]

        return {
            "conv_id" : self.conv_id[index],
            "conv_cls" : encoded_conv,
            "persona_cls" : encoded_persona,

            "encoder_hidden_state" : encoded_last.squeeze(),
            "encoder_attention_mask" : last.attention_mask,

            "decoder_input_ids" : target.input_ids.squeeze(),
            "decoder_attention_mask" : target.attention_mask.squeeze(),

            "labels" : labels.squeeze()

            "adj" : []
        }