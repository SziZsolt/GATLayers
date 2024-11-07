import torch
import torch.nn as nn

def create_adj_matrix(node_num: int, edge_index: list):
    adj_matrix = torch.zeros(size=(node_num, node_num))
    adj_matrix[edge_index[:, 0], edge_index[:, 1]] = 1
    return adj_matrix

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int, negative_slope: float, dropout: float):
        super(GraphAttentionLayer, self).__init__()
        self.out_features = out_features
        self.num_heads = num_heads
        self.W = nn.Parameter(data=torch.empty(in_features, out_features), requires_grad=True)
        self.attention = nn.Parameter(data=torch.empty(num_heads, out_features * 2, 1), requires_grad=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        h = torch.matmul(features, self.W)
        N = h.shape[0]
        h_1 = h.repeat(1, N).reshape(N * N, -1)
        h_2 = h.repeat(N, 1)
        h_3 = torch.cat([h_1, h_2], dim=-1)
        h_3 = torch.reshape(h_3, (N, -1, 2 * self.out_features))
        h_3 = h_3.unsqueeze(0).repeat(self.num_heads, 1, 1, 1)
        attention_expanded = self.attention.repeat(1, N, 1)
        h_w=(h_3 @ attention_expanded.view(self.num_heads,N,2*self.out_features,1)).squeeze(-1)
        zero_vec  = -9e15 * torch.ones_like(h_w)
        adj_expanded = adj_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)
        h_w = torch.where(adj_expanded > 0, h_w, zero_vec)
        h_w = torch.softmax(h_w, dim=-1)
        h_w = self.dropout(h_w)
        h_w = torch.mean(h_w, 0)
        out = torch.matmul(h_w, h)
        return out
    

def get_batched_data(X, y, adj, batch_size):
    X_batched = []
    y_batched = []
    adj_batched = []
    indices_batched = []
    for i in range(0, len(X), batch_size):
        X_batched.append(torch.cat(X[i:i+batch_size], dim=0))
        y_batched.append(y[i:i+batch_size])
        batch_adj_matrices = adj[i:i+batch_size]
        nodes_num = [x.shape[0] for x in batch_adj_matrices]
        result = torch.cat([torch.full((count,), i) for i, count in enumerate(nodes_num)])
        indices_batched.append(result)
        adj_batched.append(torch.block_diag(*batch_adj_matrices))
    return X_batched, y_batched, adj_batched, indices_batched