import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import random

def accuracy_fn(logits: torch.Tensor, labels:torch.Tensor):
    logits = torch.round(torch.sigmoid(logits))
    return torch.sum(logits == labels) / logits.shape[0]

def fit(model: nn.Module, num_epochs: int, patience: int, optimizer, loss_fn, X_train: torch.Tensor, edge_train: torch.Tensor, indices_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, edge_val: torch.Tensor,
        indices_val: torch.Tensor, y_val: torch.Tensor):
    best_state_dict = None
    best_loss = np.inf
    patience_counter = 0
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_acc = []
        for X_train_batch, edge_train_batch, indices_train_batch, y_train_batch in zip(X_train, edge_train, indices_train, y_train):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_batch, edge_train_batch, indices_train_batch).squeeze()
            loss = loss_fn(pred, y_train_batch)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        with torch.inference_mode():
            for X_val_batch, edge_val_batch, indices_val_batch, y_val_batch in zip(X_val, edge_val, indices_val, y_val):
                model.eval()
                pred = model(X_val_batch, edge_val_batch, indices_val_batch).squeeze()
                loss = loss_fn(pred, y_val_batch).item()
                epoch_val_loss.append(loss)
                acc = accuracy_fn(pred, y_val_batch).item()
                epoch_val_acc.append(acc)
            if loss < best_loss:
                best_loss = loss
                best_state_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stop')
                    break
        model.load_state_dict(best_state_dict)
        print(f'Epoch: {epoch + 1} | Train Loss: {np.round(np.mean(epoch_loss),4)} | Val Loss: {np.round(np.mean(epoch_val_loss),4)} | Val acc: {np.round(np.mean(epoch_val_acc),4)}')


def get_test(model: nn.Module, X: torch.Tensor, edge: torch.Tensor, indices: torch.Tensor, y: torch.Tensor):
    test_logits = []
    test_labels = []
    with torch.inference_mode():
        model.eval()
        for X_test_batch, adj_test_batch, indices_test_batch, y_test_batch in zip(X, edge, indices, y):
            pred = model(X_test_batch, adj_test_batch, indices_test_batch).squeeze()
            pred = torch.sigmoid(pred)
            pred = torch.round(pred)
            test_logits.append(pred)
            test_labels.append(y_test_batch)
    test_logits = torch.cat(test_logits).to(torch.float)
    test_labels = torch.cat(test_labels)
    return test_logits, test_labels