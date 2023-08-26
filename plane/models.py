from plane.common_imports import *
from .tools import *
from . import plane


def get_layer(config: ModelConfig):
    match config.flags_layer:
        case 'plane':
            return plane.PlaneLayer(config)
        case 'gin':
            return tgnn.GINConv(MLP(config.dim, config.dim, factor=2, drop=config.drop_out), train_eps=True)
        case 'gcn':
            return tgnn.GCNConv(config.dim, config.dim)
        case _:
            raise NotImplementedError


class ModelGraph(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_g = Embedding(config.dim_node_feature, config.dim)
        self.layers = nn.ModuleList([
            get_layer(config)
            for _ in range(config.num_layers)
        ])
        self.aggr = tgnn.SumAggregation()
        self.out_final = nn.Sequential(
            nn.LazyLinear(max(1,config.flags_mlp_factor)*config.dim),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.LazyLinear(config.dim_output),
            getattr(nn, config.act_out)()
        )

    def forward(self, data):
        num_batch = data.num_graphs
        
        # Initialize node and edge feature
        h_g = self.embed_g(data.x)

        hist = []
        for i in range(self.config.num_layers):
            match self.config.flags_layer:
                case 'plane': 
                    h_g = self.layers[i](data, h_g)
                case 'gin' | 'gcn': 
                    h_g = self.layers[i](h_g, data.edge_index)
                case _: raise NotImplementedError

            hist.append(h_g)
        
        # Aggregate all node features from i-th layer to obtain a graph-level feature
        # Concat then MLP 
        out = self.out_final(torch.cat([
            self.aggr(h_cur, data.batch, dim_size=num_batch)
            for h_cur in hist
        ], dim=1))
        return out