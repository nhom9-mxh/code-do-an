import torch
import torch_geometric.nn as gnn


class HeteroGNN(torch.nn.Module):
    edge_list = [
        ("course", "edge", "field"),
        ("course", "edge", "resource"),
        ("course", "edge", "teacher"),
        ("course", "edge", "school"),
        ("course", "edge", "user"),
        ("course", "edge", "comment"),
        ("comment", "edge", "reply"),
        ("user", "edge", "comment"),
        ("user", "edge", "reply"),
        ("school", "edge", "user"),
        ("school", "edge", "teacher"),
        ("resource", "edge", "exercise"),
        ("resource", "edge", "video"),
    ]

    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            hetero_convs = {
                edge: gnn.SAGEConv(-1, hidden_channels) for edge in HeteroGNN.edge_list
            }
            hetero_convs.update(
                {
                    (edge[2], f"rev_{edge[1]}", edge[0]): gnn.SAGEConv(
                        -1, hidden_channels
                    )
                    for edge in HeteroGNN.edge_list
                }
            )
            conv = gnn.HeteroConv(
                hetero_convs,
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = gnn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["course"])
