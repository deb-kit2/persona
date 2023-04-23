import torch
import torch.nn as nn
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"


class GCNLayer(nn.Module) :

    def __init__(self, d_in, d_out) :
        super().__init__()
        self.projection = nn.Linear(d_in, d_out)

    def forward(self, x, adj_hat) :
        # x : Node features : batch, n_nodes, d_in
        # adj_hat : adj matrix with self connections : batch, n_nodes, n_nodes

        x = self.projection(x)
        x = torch.bmm(adj_hat, x)
        x = x / adj_hat.sum(dim = -1, keepdims = True)

        return x 
    

class GCNLayerOrig(nn.Module) :

    def __init__(self, d_in, d_out) :
        super().__init__()
        self.projection = nn.Linear(d_in, d_out)

    def forward(self, x, adj_hat) :
        # x : Node features : batch, n_nodes, d_in
        # adj_hat : adj matrix with self connections : batch, n_nodes, n_nodes

        n_nodes = adj_hat.size()[1]
        adj = adj_hat - torch.eye(n_nodes).to(device) # without self connections

        d_hat = adj.sum(dim = -1)
        d_hat = torch.pow(d_hat, -0.5)
        d_hat = torch.diag_embed(d_hat) # batch, n_nodes, n_nodes

        dad = torch.bmm(torch.bmm(d_hat, adj_hat), d_hat) # normalizing matrix

        x = self.projection(x) # to another dimension
        x = torch.bmm(dad, x) # for all node embeddings, in a matrix form

        return x 
    

class GATLayer(nn.Module) :

    def __init__(self, d_in, d_out, n_heads = 1, concat_heads = True, alpha = 0.2) :
        super().__init__()
        
        self.n_heads = n_heads
        self.concat_heads = concat_heads
        self.d_out = d_out
        if concat_heads :
            assert d_out % n_heads == 0
            self.d_out = d_out // n_heads

        self.projection = nn.Linear(d_in, d_out * n_heads)
        self.a = nn.Parameter(torch.Tensor(n_heads, 2 * d_out))
        self.leakyRelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim = -2)

        # from the original paper
        nn.init.xavier_uniform_(self.projection.weight.data, gain = 1.414)
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)

    def forward(self, x, adj_hat, return_attentions = False) :
        # x : Node features : batch_size, n_nodes, d_in
        # adj_hat : adj matrix with self connections : batch_size, n_nodes, n_nodes

        B, N = x.size()[ : 2]

        x = self.projection(x)
        x = x.view(B, N, self.n_heads, self.d_out)

        # p.shape : B, N x N, n_heads, 2 x d_out
        p1 = x.repeat_interleave(N, dim = 1)
        p2 = x.repeat(1, N, 1, 1)
        p = torch.cat([p1, p2], dim = -1)
        p = p.view(B, N, N, self.n_heads, 2 * self.d_out)

        e = torch.einsum("bpqhd, hd -> bpqh", p, self.a)

        e = self.leakyRelu(e)

        # where there is no connection, att = 0
        e = torch.where(adj_hat.unsqueeze(-1) == 0, float("-inf"), e)

        attentions = self.softmax(e)
        res = torch.einsum("bmnh, bnhd -> bmhd", attentions, x)

        if self.concat_heads :
            res = res.reshape(B, N, self.n_heads * self.d_out)
        else :
            res = res.mean(dim = -1)

        if return_attentions :
            return res, attentions
        return res
    