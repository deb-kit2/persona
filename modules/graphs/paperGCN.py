import torch
import torch.nn as nn
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    

# test
def main() :
    # batch, n_nodes, n_nodes
    adj_hat = torch.Tensor([[[1, 1, 0, 0],
                             [1, 1, 1, 1],
                             [0, 1, 1, 1],
                             [0, 1, 1, 1]]])

    # batch, n_node, d
    node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)

    print("Node features :\n", node_feats)

    # with projection == identity matrix
    # need to make test


if __name__ == "__main__" :
    main()