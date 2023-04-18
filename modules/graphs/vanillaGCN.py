import torch
import torch.nn as nn
import torch.nn.functional as f


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
    
# test
def main() :
    # batch, n_nodes, n_nodes
    adj_hat = torch.Tensor([[[1, 1, 0, 0],
                             [1, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]]])

    # batch, n_node, d
    node_feats = torch.arange(8, dtype = torch.float32).view(1, 4, 2)

    print("Node features :\n", node_feats)

    # with projection == identity matrix
    # expected : self + adjacent nodes' embeddings
    # normalized by the number of neighbors + 1

    layer = GCNLayer(d_in = 2, d_out = 2)

    layer.projection.weight.data = torch.Tensor([
        [1., 0.],
        [0., 1.]
    ])
    layer.projection.bias.data = torch.Tensor([[0., 0.]])

    with torch.no_grad() :
        output = layer(node_feats, adj_hat)

    print("Output features : \n", output)


if __name__ == "__main__" :
    main()