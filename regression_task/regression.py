import torch
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
import random


def create_node_features(len_dataset: int, num_nodes: list[int], max_nodes: int, num_features: int):
    node_features = torch.zeros(size=(len_dataset, max_nodes, num_features))
    for i, num_node in enumerate(num_nodes):
        features = torch.zeros(size=(max_nodes, num_features))
        features[:num_node] = torch.rand(size=(num_node, num_features))
        node_features[i] = features
    node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)
    return node_features


class MetricsTracker():
    def __init__(self, loss_fn: nn.Module, patience: int):
        self.train_metrics = {
            'loss': [],
            'mae': [],
            'rmse': [],
        }
        self.val_metrics = copy.deepcopy(self.train_metrics)
        self.temp_train_metrics = copy.deepcopy(self.train_metrics)
        self.temp_val_metrics = copy.deepcopy(self.train_metrics)
        self.loss_fn = loss_fn
        self.best_val_loss = np.inf
        self.patience = patience
        self.patience_counter = 0

    def step(self, logits: torch.Tensor, labels: torch.Tensor, val: bool = False):
        loss, mae, rmse = self.get_metrics(logits, labels)
        metrics = self.temp_val_metrics if val else self.temp_train_metrics
        metrics['loss'].append(loss)
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        return loss

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.squeeze()
        labels = labels.squeeze()
        loss = self.loss_fn(logits, labels)
        mae_error = MeanAbsoluteError()
        mse_error = MeanSquaredError()
        mae = mae_error(logits, labels)
        rmse = torch.sqrt(mse_error(logits, labels))
        return loss, mae, rmse
    
    def end_epoch(self, num_epoch: int):
        self.__refresh(self.train_metrics, self.temp_train_metrics)
        self.__refresh(self.val_metrics, self.temp_val_metrics)
        loss = self.train_metrics['loss'][-1]
        val_loss = self.val_metrics['loss'][-1]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        print(f'Epoch: {num_epoch} | Train Loss: {loss:3f} | Val Loss: {val_loss:3f}')
        
    
    def is_early_stop(self):
        return self.patience_counter >= self.patience
    
    def __refresh(self, metrics, temp_metrics):
        metrics['loss'].append(np.mean(temp_metrics['loss']))
        metrics['mae'].append(np.mean(temp_metrics['mae']))
        metrics['rmse'].append(np.mean(temp_metrics['rmse']))
        for k in temp_metrics:
            temp_metrics[k] = []

    def plot_history(self):
        keys = [k for k in self.train_metrics]
        fig, axes = plt.subplots(1, len(keys), figsize=(20,6))
        for i, k in enumerate(keys):
            axes[i].plot(self.train_metrics[k], label='Train')
            axes[i].plot(self.val_metrics[k], label='Validation')
            axes[i].grid()
            axes[i].set_title(k.upper())
            axes[i].legend()


def fit(model: nn.Module, optimizer, metrics_tracker: MetricsTracker, num_epochs: int, X_train: torch.Tensor, edge_train: torch.Tensor,
        indices_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, edge_val: torch.Tensor, indices_val: torch.Tensor,
        y_val: torch.Tensor):
    for epoch in range(num_epochs):
        for X_train_batch, edge_train_batch, indices_train_batch, y_train_batch in zip(X_train, edge_train, indices_train, y_train):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_batch, edge_train_batch, indices_train_batch).squeeze()
            loss = metrics_tracker.step(pred, y_train_batch)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            for X_val_batch, edge_val_batch, indices_val_batch, y_val_batch in zip(X_val, edge_val, indices_val, y_val):
                model.eval()
                pred = model(X_val_batch, edge_val_batch, indices_val_batch).squeeze()
                metrics_tracker.step(pred, y_val_batch, True)
            metrics_tracker.end_epoch(epoch)
            if metrics_tracker.is_early_stop():
                print('Early Stop')
                break


def plot_tests(test_logits: torch.Tensor, test_labels: torch.Tensor):
    indices = random.sample(list(range(len(test_logits))), 5)
    rows = 2
    cols = test_labels.shape[-1] // rows
    fig, axes = plt.subplots(rows, cols, figsize=(20,6))
    counter = 0
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            axes[i,j].plot(test_logits[indices, k], label='Pred')
            axes[i,j].plot(test_labels[indices, k], label='Real')
            axes[i,j].grid()
            axes[i,j].legend()
    fig.suptitle('Samples on test data on every feature')
    plt.show()