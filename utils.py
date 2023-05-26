import json

import torch
from torch.utils.data import Dataset


class PersonaDataset(Dataset) :
    # Dataset class for the PersonaChat dataset

    def __init__(self, args, subset = "train") :
        name = "data/" + args.encdec + "." + args.pretrained_name.split("-")[-1]
        name += "." + args.encrep + "." + args.graph_type + "." + subset + ".json"

        with open(name, "r", encoding = "utf-8") as fi :
            self.data = json.load(fi)

    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, index) :
        data = self.data[index]
        data = {k : torch.tensor(v) for k, v in data.items()}
        
        return data
