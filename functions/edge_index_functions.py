import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn

def get_batched_data(X, y, edge_indices, batch_size):
    X_batched = []
    y_batched = []
    edge_index_batched = []
    indices_batched = []
    
    for i in range(0, len(X), batch_size):
        node_offset = 0
        X_batched.append(torch.cat(X[i:i+batch_size], dim=0))
        y_batched.append(y[i:i+batch_size])
        
        batch_edge_indices = edge_indices[i:i+batch_size]
        node_sizes = [x.shape[0] for x in X[i:i+batch_size]]
        
        edge_index_batch = []
        for j, edge_index in enumerate(batch_edge_indices):
            edge_index_batch.append(edge_index + node_offset)
            node_offset += node_sizes[j]
            
        edge_index_batched.append(torch.cat(edge_index_batch, dim=1))
        result = torch.cat([torch.full((count,), idx) for idx, count in enumerate(node_sizes)])
        indices_batched.append(result)

    return X_batched, y_batched, edge_index_batched, indices_batched

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, num_heads: int, negative_slope: float, dropout: float):
        super(GraphAttentionLayer, self).__init__(aggr='add')
        self.out_features = out_features
        self.num_heads = num_heads
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.attention = nn.Parameter(torch.Tensor(num_heads, out_features * 2, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = torch.matmul(x, self.W)
        out = self.propagate(edge_index, h=h)
        return out

    def message(self, h_i, h_j, edge_index_i):
        # pairing the source - target nodes
        h_cat = torch.cat([h_i, h_j], dim=-1)
        # applying attention
        h_attention = h_cat @ self.attention
        h_attention = h_attention.squeeze(-1)
        # Here the idea is to flatten and need to offset the edge indices to handle the num_heads dimensions separately
        edge_index_i_expanded = edge_index_i.repeat(self.num_heads) + (torch.arange(self.num_heads) * h_attention.shape[1]).repeat_interleave(edge_index_i.shape[0])
        alpha = softmax(h_attention.view(-1), edge_index_i_expanded)
        alpha = self.dropout(alpha)
        alpha = alpha.view(self.num_heads, -1).mean(0)  
        return h_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out
