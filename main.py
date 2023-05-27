import os
import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f

from model import BartForPersonaAwareGeneration
from utils import PersonaDataset

from torch.utils.data import DataLoader

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
    parser.add_argument("--save_name", type = str, required = True, help = "models/trial_1.pt")
    
    parser.add_argument("--print_every", type = int, default = 1)

    args = parser.parse_args()
    return args


def product(p) :
    res = 1
    for l in p.size() :
        res  *= l
    return res


def train_step(model, data, optimizer, scheduler, loss_function) :
    
    model.train()
    for batch in data :
        batch = {k : v.to(device) for k, v in batch.items()}

        logits = model(x = batch["conv_cls"], persona = batch["persona_cls"], 
                       adj_hat = batch["adj"], mask = batch["conv_mask"],
                       encoder_hidden_states = batch["encoder_hidden_states"], 
                       encoder_attention_mask = batch["encoder_attention_mask"],
                       decoder_input_ids = batch["decoder_input_ids"], 
                       decoder_attention_mask = batch["decoder_attention_mask"])
        
        loss = loss_function(logits.view(-1, model.decoder.config.vocab_size), batch["labels"].view(-1))
        acc = torch.sum(torch.argmax(logits, -1) == batch["labels"]) / product(batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss.item(), acc.item()


@torch.no_grad()
def evaluate_step(model, data, loss_function) :

    model.eval()
    for batch in data :
        batch = {k : v.to(device) for k, v in batch.items()}

        logits = model(x = batch["conv_cls"], persona = batch["persona_cls"], 
                       adj_hat = batch["adj"], mask = batch["conv_mask"],
                       encoder_hidden_states = batch["encoder_hidden_states"], 
                       encoder_attention_mask = batch["encoder_attention_mask"],
                       decoder_input_ids = batch["decoder_input_ids"], 
                       decoder_attention_mask = batch["decoder_attention_mask"])
        
        loss = loss_function(logits.view(-1, model.decoder.config.vocab_size), batch["labels"].view(-1))
        acc = torch.sum(torch.argmax(logits, -1) == batch["labels"]) / product(batch["labels"])

    return loss.item(), acc.item()


def train(model, train_data, test_data,
          optimizer, scheduler = None,
          loss_function = nn.CrossEntropyLoss(), 
          max_epochs = 200,
          early_stopping = 10,
          print_every = 20, 
          save_best = True, save_name = "models/ourModel") :

    logging.info("Model training started. :D\n")
    history = {"loss" : [], "val_loss" : [], "acc" : [], "val_acc" : []}
    best_val_loss = 9e15

    for epoch in range(max_epochs) :
        loss, acc = train_step(model, train_data, optimizer, scheduler, loss_function)
        val_loss, val_acc = evaluate_step(model, test_data, loss_function)

        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        if epoch % print_every == 0 :
            logging.debug(f"Epoch : {epoch}\n##########" + \
                          f"\nTrain loss : {loss:.4f}, Train accuracy : {acc:.4f}" + \
                          f"\nValidation loss : {val_loss:.4f}, Validation accuracy : {val_acc:.4f}")
        
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            if save_best :
                logging.info("Saving best model : " + save_name + ".pt")
                torch.save(model.state_dict(), save_name + ".pt")
        
        if epoch > early_stopping and val_loss > sum(history["val_loss"][-early_stopping -1 : -1]) / early_stopping :
            logging.info("\nEarly stopping...")
            break

    return history


def someClass(args) : 
    return

if __name__ == "__main__" :
    # reproducibility
    torch.manual_seed(42)

    args = parse_args()

    if not os.path.isdir("models") :
        os.mkdir("models")
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
        output_dir = "models/",
        do_train = args.train,
        do_eval = args.test,
        
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate = args.lr,
        weight_decay = args.weight_decay,
        num_train_epochs = args.epochs,
        warmup_steps = args.warmup,
        
        evaluation_strategy = "epoch",
        eval_accumulation_steps = 10,

        log_level = "debug",
        logging_dir = "logs/",
        logging_strategy = "epoch",

        save_strategy = "epoch",
        save_total_limit = 2,
        report_to = "none"
    )
    logging.info("Training args ready.")
    
    if args.train : 
        train_data = PersonaDataset(args, "train")
        train_data = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
        logging.info("Train data loaded.")

    test_data = PersonaDataset(args, "valid")
    test_data = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)
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
        
        trainer.save_model("models/")
        logging.info("Saved the model at \"models/\"")

    else :
        raise NotImplementedError("Testing in main not yet supported.")
    