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

    # model args
    parser.add_argument("--encdec", type = str, default = "bart")
    # callbacks
    parser.add_argument("--es", type = int, default = 10)
    parser.add_argument("--save_best", type = bool, default = True)
    

    args = parser.parse_args()
    return args


def train_step(model, data, optimizer, loss_function) :
    
    model.train()
    # do something

    loss = loss_function(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = torch.sum(preds == y) / y.shape[0]

    return loss.item(), accuracy.item()


@torch.no_grad()
def evaluate_step(model, data, loss_function) :

    model.eval()
    # do something

    preds = logits.argmax(dim = -1)
    y = node_labels[mask]

    loss = loss_function(logits, y)
    accuracy = torch.sum(preds == y) / y.shape[0]

    return loss.item(), accuracy.item()


def train(model, data, optimizer, train_mask, val_mask, adj = None,
          loss_function = nn.CrossEntropyLoss(), 
          max_epochs = 200,
          early_stopping = 10,
          print_every = 20, 
          save_best = True, save_name = "models/ourModel") :

    history = {"loss" : [], "val_loss" : [], "acc" : [], "val_acc" : []}
    best_val_loss = 9e15

    for epoch in range(max_epochs) :
        loss, acc = train_step(model, data, optimizer, loss_function, 
                               mask = train_mask, adj = adj)
        val_loss, val_acc = evaluate_step(model, data, loss_function, 
                                          mask = val_mask, adj = adj)

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
    