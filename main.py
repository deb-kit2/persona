import os
import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f

from model import BartForPersonaAwareGeneration
from utils import PersonaDataset

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() :
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument("--train", type = bool, default = True)
    parser.add_argument("--lr", type = float, default = 5e-5)
    parser.add_argument("--weight_decay", type = float, default = 0.01)
    parser.add_argument("--epochs", type = int, default = 200)
    parser.add_argument("--max_conv_length", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--warmup", type = int, default = 2)

    # model args
    parser.add_argument("--max_length", type = int, default = 32)
    parser.add_argument("--encdec", type = str, default = "bart")
    parser.add_argument("--pretrained_name", type = str, default = "facebook/bart-base")
    parser.add_argument("--graph_type", type = str, default = "paperGCN")
    parser.add_argument("--encrep", type = str, default = "first",
                        help = "can be 'mean' or 'first'")
    parser.add_argument("--num_heads", type = int, default = 1)
    parser.add_argument("--dropout",type = float, default = 0.25)

    # callbacks
    parser.add_argument("--es", type = int, default = 10)
    parser.add_argument("--save_best", type = bool, default = True)
    parser.add_argument("--save_name", type = str, required = True, help = "models/trial_1")
    
    parser.add_argument("--print_every", type = int, default = 1)

    args = parser.parse_args()
    return args


if __name__ == "__main__" :
    # reproducibility
    torch.manual_seed(42)

    args = parse_args()

    if not os.path.isdir("models") :
        os.mkdir("models")
    if not os.path.isdir(args.save_name) :
        os.mkdir(args.save_name)
    if not os.path.isdir("logs") :
        os.mkdir("logs")

    logging.basicConfig(level = logging.DEBUG,
                        handlers = [
                            logging.FileHandler("logs/" + args.save_name.split("/")[-1] + ".log"),
                            logging.StreamHandler()
                        ],
                        format = "%(levelname)s : %(message)s")
    
    logging.info("\n" + json.dumps({key : value for key, value in vars(args).items()}, indent = 4) + "\n")
    

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.save_name,
        do_train = args.train,
        do_eval = True,
        
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate = args.lr,
        weight_decay = args.weight_decay,
        num_train_epochs = args.epochs,
        warmup_steps = args.warmup,
        
        evaluation_strategy = "epoch",
        eval_accumulation_steps = 10,

        log_level = "debug",
        logging_dir = "logs",
        logging_strategy = "epoch",

        save_strategy = "epoch",
        save_total_limit = 2,
        report_to = "none"
    )
    logging.info("Training args ready.")
    
    
    if args.train : 
        train_data = PersonaDataset(args, "train")
        logging.info("Train data loaded.")

    test_data = PersonaDataset(args, "valid")
    logging.info("Test data loaded.")

    model = BartForPersonaAwareGeneration.from_pretrained(args.pretrained_name, args)
    logging.info("Model initialized.") 
    
    
    if args.train :
        trainer = Seq2SeqTrainer(
            model = model,
            args = training_args,
            train_dataset = train_data,
            eval_dataset = test_data,
        )
        logging.info("Trainer ready. Starting model training...")
        
        trainer.train()
        logging.info("Model training finished.")
        
        trainer.save_model(args.savename)
        logging.info(f"Saved the model at {args.save_name}")

    else :
        raise NotImplementedError("Testing in main not yet supported.")
