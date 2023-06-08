import json

import torch
from torch.utils.data import Dataset


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
        name = "data/" + args.encdec + "." + args.pretrained_name.split("-")[-1]
        name += "." + args.encrep + "." + subset + ".json"

        with open(name, "r", encoding = "utf-8") as fi :
            self.data = json.load(fi)

    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, index) :
        data = self.data[index]
        data = {k : torch.tensor(v) for k, v in data.items()}
        
        data["decoder_input_ids"] = data["decoder_input_ids"].long()
        data["adj"] = construct_adj(data["conv_lens"], data["persona_lens"])

        return data
