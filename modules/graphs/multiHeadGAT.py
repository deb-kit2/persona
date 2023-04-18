import torch
import torch.nn as nn
import torch.nn.functional as f


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
    

# for test
def main() :
    layer = GATLayer(2, 2, n_heads = 2)
    layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
    layer.projection.bias.data = torch.Tensor([0., 0.])
    layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

    adj_matrix = torch.Tensor([
        [[1., 1., 0., 0.],
         [1., 1., 1., 1.],
         [0., 1., 1., 1.],
         [0., 1., 1., 1.]],
        
        [[1., 1., 0., 0.],
         [1., 1., 1., 0.],
         [0., 1., 1., 0.],
         [0., 0., 0., 1.]]
        ])

    # node_feats = torch.arange(8, dtype = torch.float32).view(1, 4, 2)
    node_feats = torch.Tensor(
        [[[0., 1.],
          [2., 3.],
          [4., 5.],
          [6., 7.]],
        
        [[8., 9.],
         [10., 11.],
         [12., 13.],
         [0., 0.]]
        ])
    
    with torch.no_grad() :
        out_feats, attentions = layer(node_feats, adj_matrix, True)
        print("Output node features : ")
        print(out_feats, "\n")

        print("Attentions : ")
        print(attentions.permute(0, 3, 1, 2))

        # need to calculate by hand and make equality checks

if __name__ == "__main__" :
    main()