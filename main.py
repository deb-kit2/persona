import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f


def parse_args() :
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument("--train", type = bool, default = True)
    parser.add_argument("--lr", type = float, default = 3e-4)
    parser.add_argument("--epochs", type = int, default = 200)
    parser.add_argument("--max_conv_length", type = int, default = 50)

    # model args
    parser.add_argument("--max_length", type = int, default = 32)
    parser.add_argument("--encdec", type = str, default = "bart")
    parser.add_argument("--pretrained_name", type = str, default = "facebook/bart-base")
    parser.add_argument("--graph_type", type = str, default = "paperGCN")
    parser.add_argument("--encrep", type = "str", default = "first",
                        help = "can be 'mean' or 'first'")
    parser.add_argument("--heads", type = int, default = 8)
    parser.add_argument("--dropout",type = float, default = 0.25)

    # callbacks
    parser.add_argument("--es", type = int, default = 10)
    parser.add_argument("--save_best", type = bool, default = True)
    parser.add_argument("--save_name", type = str, default = "models/")
    
    args = parser.parse_args()
    return args


def train_step(model, data, optimizer, loss_function) :
    
    model.train()
    # do something

    return


@torch.no_grad()
def evaluate_step(model, data, loss_function) :

    model.eval()
    # do something
    
    return


def train(model, data, optimizer, train_mask, val_mask, adj = None,
          loss_function = nn.CrossEntropyLoss(), 
          max_epochs = 200,
          early_stopping = 10,
          print_every = 20, 
          save_best = True, save_name = "models/ourModel") :

    history = {"loss" : [], "val_loss" : [], "acc" : [], "val_acc" : []}
    best_val_loss = 9e15

    for epoch in range(max_epochs) :
        loss, acc = train_step(model, data, optimizer, loss_function)
        val_loss, val_acc = evaluate_step(model, data, loss_function)

        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            if save_best :
                torch.save(model.state_dict(), save_name + f"_{epoch}.pt")

        if epoch > early_stopping and val_loss > sum(history["val_loss"][-early_stopping -1 : -1]) / early_stopping :
            print("\nEarly stopping...")
            break

        if epoch % print_every == 0 :
            print(f"\nEpoch : {epoch}\n##########")
            print(f"Train loss : {loss:.4f}, Train accuracy : {acc:.4f}")
            print(f"Validation loss : {val_loss:.4f}, Validation accuracy : {val_acc:.4f}")

    return history


if __name__ == "__main__" :
    args = parse_args()
    