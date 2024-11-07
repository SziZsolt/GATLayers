import random
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

def get_indices(length: int, train_size: float, val_size: float):
    rand_indices = list(range(0, length))
    random.shuffle(rand_indices)
    train_indices = (0, int(length * train_size))
    val_indices = (train_indices[1], train_indices[1]+int(length * val_size))
    test_indices = (val_indices[1], length)
    train_indices = rand_indices[train_indices[0]:train_indices[1]]
    val_indices = rand_indices[val_indices[0]:val_indices[1]]
    test_indices = rand_indices[test_indices[0]:test_indices[1]]
    return train_indices, val_indices, test_indices

def get_test(model: nn.Module, X: torch.Tensor, edge: torch.Tensor, indices: torch.Tensor, y: torch.Tensor):
    test_logits = []
    test_labels = []
    with torch.inference_mode():
        model.eval()
        for X_test_batch, adj_test_batch, indices_test_batch, y_test_batch in zip(X, edge, indices, y):
            pred = model(X_test_batch, adj_test_batch, indices_test_batch).squeeze()
            test_logits.append(pred)
            test_labels.append(y_test_batch)
    test_logits = torch.cat(test_logits).to(torch.float)
    test_labels = torch.cat(test_labels)
    return test_logits, test_labels


def get_diameters(dataset):
    diameters = torch.zeros(size=(len(dataset),))
    for i, data in enumerate(dataset):
        G = to_networkx(data)
        try:
            diameter = nx.diameter(G)
        except:
            diameter = -1
        diameters[i] = diameter
    return diameters

def set_all_seeds(seed: int):
    seed = 42
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (for single GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (for all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility