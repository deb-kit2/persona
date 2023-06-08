import os
import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f

from model import PersonaAwareGeneration
from utils import PersonaDataset

from torch.utils.data import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() :
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument("--train", type = bool, default = True)
    parser.add_argument("--lr", type = float, default = 3e-4)
    parser.add_argument("--epochs", type = int, default = 50)
    parser.add_argument("--max_conv_length", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--warmup", type = int, default = 3)

    # vocab and embedding
    parser.add_argument("--vocab_size", type = int, default = 50265)
    parser.add_argument("--embedding_path", type = str, default = "data/embedding.table.bart.base.pt")

    # model args
    parser.add_argument("--max_length", type = int, default = 32)
    parser.add_argument("--encdec", type = str, default = "bart")
    parser.add_argument("--pretrained_name", type = str, default = "facebook/bart-base")
    parser.add_argument("--graph_type", type = str, default = "paperGCN")
    parser.add_argument("--encrep", type = str, default = "first",
                        help = "can be 'mean' or 'first'")
    parser.add_argument("--num_heads", type = int, default = 2)
    parser.add_argument("--dropout",type = float, default = 0.125)

    # callbacks
    parser.add_argument("--es", type = int, default = 10)
    parser.add_argument("--save_best", type = bool, default = True)
    parser.add_argument("--save_name", type = str, required = True)
    
    parser.add_argument("--print_every", type = int, default = 1)

    args = parser.parse_args()
    return args


def product(p) :
    res = 1
    for l in p.size() :
        res  *= l
    return res


def train_step(model, data, optimizer, scheduler, loss_function, vocab_size = 50265) :
    
    model.train()
    for batch in data :
        batch = {k : v.to(device) for k, v in batch.items()}

        logits = model(conv_cls = batch["conv_cls"], persona_cls = batch["persona_cls"], 
                       adj = batch["adj"], conv_mask = batch["conv_mask"],
                       h_enc_mask = batch["h_enc_mask"],
                       decoder_input_ids = batch["decoder_input_ids"], 
                       decoder_attention_mask = batch["decoder_attention_mask"])
        
        logits = logits["logits"]
        loss = loss_function(logits.view(-1, vocab_size), batch["labels"].view(-1))
        acc = torch.sum(torch.argmax(logits, -1) == batch["labels"]) / product(batch["labels"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss.item(), acc.item()


@torch.no_grad()
def evaluate_step(model, data, loss_function, vocab_size = 50265) :

    model.eval()
    for batch in data :
        batch = {k : v.to(device) for k, v in batch.items()}

        logits = model(conv_cls = batch["conv_cls"], persona_cls = batch["persona_cls"], 
                       adj = batch["adj"], conv_mask = batch["conv_mask"],
                       h_enc_mask = batch["h_enc_mask"],
                       decoder_input_ids = batch["decoder_input_ids"], 
                       decoder_attention_mask = batch["decoder_attention_mask"])
        
        logits = logits["logits"]
        loss = loss_function(logits.view(-1, vocab_size), batch["labels"].view(-1))
        acc = torch.sum(torch.argmax(logits, -1) == batch["labels"]) / product(batch["labels"])

    return loss.item(), acc.item()


def train(model, train_data, test_data,
          optimizer, scheduler = None, vocab_size = 50265,
          loss_function = nn.CrossEntropyLoss(), 
          max_epochs = 200,
          early_stopping = 10,
          print_every = 20, 
          save_best = True, save_name = "models/ourModel") :
    
    logging.info("Model training started. :D\n")
    history = {"loss" : [], "val_loss" : [], "acc" : [], "val_acc" : []}
    best_val_loss = 9e15

    for epoch in range(max_epochs) :
        loss, acc = train_step(model, train_data, optimizer, scheduler, loss_function, vocab_size)
        val_loss, val_acc = evaluate_step(model, test_data, loss_function, vocab_size)

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
    
    if args.train : 
        train_data = PersonaDataset(args, "train")
        train_data = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
        logging.info("Train data loaded.")

    test_data = PersonaDataset(args, "valid")
    test_data = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)
    logging.info("Test data loaded.")

    model = PersonaAwareGeneration(args).to(device)
    logging.info("Model initialized.") 
    
    if args.train :
        optimizer = AdamW(model.parameters(), lr = args.lr)
        scheduler = LinearLR(optimizer, total_iters = args.warmup * len(train_data))
        logging.info("Optimizer and Scheduler ready.")
        
        history = train(model, train_data, test_data,
                        optimizer = optimizer, scheduler = scheduler,
                        max_epochs = args.epochs, early_stopping = args.es, 
                        print_every = args.print_every, save_name = args.save_name)
        logging.info("Model training finished.")

    else :
        model.load_state_dict(args.save_name)
        logging.info("Model weights loaded.")

        loss, acc = evaluate_step(model, test_data, nn.CrossEntropyLoss())
        logging.info(f"Model f{args.save_name}\n##########\nValidation loss : {loss:.4f}, Validation accuracy : {acc:.4f}")
        