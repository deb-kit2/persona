import sys
import argparse
import copy
import json

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

sys.path.append("..")
from transformers import BartTokenizer, T5Tokenizer
from modules.encoders import BARTEncoder, T5Encoder


subset = "valid"
encdec = "bart"
pretrained_name = "facebook/bart-base"
d_in = 768
encrep = "first"
graph_type = "paperGCN"

max_len = 32
max_conv_length = 50
max_persona_length = 6

conv_id = []
conv_lens = [] # len - 1

persona = []
conversation = [] # everything except target
last = [] # last in conversation, <CLS> to be merged here
target = [] # to be produced

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() :
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--max_conv_length", type = int, default = 50)
    parser.add_argument("--subset", type = str, required = True)

    # model args
    parser.add_argument("--max_length", type = int, default = 32)
    parser.add_argument("--encdec", type = str, default = "bart")
    parser.add_argument("--pretrained_name", type = str, default = "facebook/bart-base")
    parser.add_argument("--encrep", type = str, default = "first",
                        help = "can be 'mean' or 'first'")
    parser.add_argument("--graph_type", type = str, default = "paperGCN")

    args = parser.parse_args()
    return args

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

def __getitem__(index) :

    con = tokenizer.batch_encode_plus(
        conversation[index],
        max_length = max_len,
        truncation = True, 
        padding = "max_length",
        return_tensors = "pt",
        return_attention_mask = True,
        verbose = False
    ).to(device)
    encoded_conv = encoder(con.input_ids, con.attention_mask)[0]
    if encrep == "first" :
        encoded_conv = encoded_conv[:, 0, :]
    else :
        encoded_conv = torch.mean(encoded_conv, dim = -2)

    dummy = torch.zeros((max_conv_length - conv_lens[index], d_in), dtype = torch.float32).to(device)
    encoded_conv = torch.cat((dummy, encoded_conv), dim = 0)

    mask = [True] * (max_conv_length - conv_lens[index]) + [False] * conv_lens[index]
    mask = torch.tensor(mask, dtype = torch.bool)

    tar = tokenizer.batch_encode_plus(
        [target[index]],
        max_length = max_len,
        truncation = True, 
        padding = "max_length",
        return_tensors = "pt",
        return_attention_mask = True,
        verbose = False
    ).to(device)

    labels = copy.deepcopy(tar.input_ids)
    labels = torch.tensor([[-100 if token == tokenizer.pad_token_id else token for token in l] for l in labels])

    per = tokenizer.batch_encode_plus(
        persona[index],
        max_length = max_len,
        truncation = True,
        padding = "max_length",
        return_tensors = "pt",
        return_attention_mask = True,
        verbose = False
    ).to(device)
    encoded_persona = encoder(per.input_ids, per.attention_mask)[0]
    if encrep == "first" :
        encoded_persona = encoded_persona[:, 0, :]
    else :
        encoded_persona = torch.mean(encoded_persona, dim = -2)

    dummy = torch.zeros((max_persona_length - len(persona[index]), d_in), dtype = torch.float32).to(device)
    encoded_persona = torch.cat((encoded_persona, dummy), dim = 0)

    las = tokenizer.batch_encode_plus(
        [last[index]],
        max_length = max_len,
        truncation = True,
        padding = "max_length",
        return_tensors = "pt",
        return_attention_mask = True,
        verbose = False
    ).to(device)
    encoded_last = encoder(las.input_ids, las.attention_mask)[0]

    if graph_type == "paperGCN" :
        adj = construct_adj(conv_lens[index], len(persona[index]))
    else :
        raise NotImplementedError("Not implemetned others.")
    
    return {
        "conv_id" : conv_id[index],
        "conv_cls" : encoded_conv.tolist(),
        "conv_mask" : mask.tolist(),
        
        "persona_cls" : encoded_persona.tolist(),

        "encoder_hidden_states" : encoded_last.squeeze().tolist(),
        "encoder_attention_mask" : las.attention_mask.squeeze().tolist(),

        "decoder_input_ids" : tar.input_ids.squeeze().tolist(),
        "decoder_attention_mask" : tar.attention_mask.squeeze().tolist(),

        "labels" : labels.squeeze().tolist(),

        "adj" : adj.tolist()
    }


if __name__ == "__main__" :
    args = parse_args()

    subset = args.subset
    encdec = args.encdec
    pretrained_name = args.pretrained_name
    encrep = args.encrep
    graph_type = args.graph_type
    
    if args.pretrained_name.endswith("base") :
        d_in = 768
    elif args.pretrained_name.endswith("large") :
        d_in = 1024
    else :
        d_in = 512

    max_len = args.max_length
    max_conv_length = args.max_conv_length

    if encdec == "bart" :
        # init bart
        tokenizer = BartTokenizer.from_pretrained(pretrained_name)
        encoder = BARTEncoder.from_pretrained(pretrained_name).eval()

        encoder.to(device)
    else :
        raise NotImplementedError()

    DATA_PATH = "personachat_data.json"
    with open(DATA_PATH, "r", encoding = "utf-8") as fi :
        data = json.load(fi)[subset]

        for sample in data :
            conv_id.append(sample["conv_id"])
            conv_lens.append(len(sample["dialog"]) - 1)

            persona.append(sample["personality_person2"])

            conversation.append([sample["dialog"][p]["text"] for p in range(len(sample["dialog"]) - 1)])
            last.append(sample["dialog"][-2]["text"])

            target.append(sample["dialog"][-1]["text"])

        del data

    name = pretrained_name.split("/")[-1]
    name = ".".join(name.split("-") + [encrep, subset, "json"])

    writer = open(name, "w", encoding = "utf-8")
    writer.write("[\n")

    for i in tqdm(range(len(conv_id))) :
        x = __getitem__(i)
        writer.write(json.dumps(x) + "\n")

    writer.write("]\n")
    writer.close()
