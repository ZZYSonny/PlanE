from plane.common_imports import *
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

@dataclass
class ModelConfig:
    dim_output: int
    dim_node_feature: typing.Union[int, str, list]
    dim_edge_feature: typing.Union[int, str, list, None]

    dim: int = 64
    dim_plane_pe: int = 0

    num_layers: int = 2
    act_out: str = "Identity"
    flags_layer: str = "plane"

    drop_enc: float = 0
    drop_rec: float = 0
    drop_agg: float = 0
    drop_com: float = 0
    drop_out: float = 0
    drop_edg: float = 0

    flags_plane_agg: str = "n_t_b_gr_pr"
    flags_norm_before_com: str = "batch_norm"
    flags_mlp_factor: int = 2
    flags_mlp_layer: int = 2
    flags_plane_gine_type: str = "incomplete"

    def __post_init__(self):
        self.plane_terms = self.flags_plane_agg.split("_")

def MLP(dim_in, dim_out=None, factor=-1, drop=0.0, norm='batch_norm'):
    if dim_out is None: dim_out = dim_in
    if factor == -1:
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.Identity() if norm == "None" else nn.BatchNorm1d(dim_out),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    else:
        return nn.Sequential(
            nn.Linear(dim_in, dim_in*factor),
            nn.Identity() if norm == "None" else nn.BatchNorm1d(dim_in*factor),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dim_in*factor, dim_out),
            nn.Identity() if norm == "None" else nn.BatchNorm1d(dim_out),
            nn.ReLU(),
            nn.Dropout(drop)
        )

def SANDWICH(dim, p):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.ReLU(),
        nn.Dropout(p)
    )

class GINE(tgnn.models.basic_gnn.BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        mlp = tgnn.models.MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return tgnn.GINEConv(mlp, **kwargs)

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        match input_dim:
            case "ogb_atom_node": self.embed = AtomEncoder(output_dim)
            case "ogb_atom_edge": self.embed = BondEncoder(output_dim)
            case "lin":           self.embed = nn.LazyLinear(output_dim)    
            case x if isinstance(x, int):
                self.embed = nn.Embedding(x, output_dim)
            case [x] if isinstance(x, int): 
                self.embed1 = nn.Embedding(x, output_dim)
                self.embed = lambda x: self.embed1(x.flatten())
            case "None":
                self.embed = lambda x: None
            case _:
                raise NotImplementedError()
    
    def forward(self, data):
        return self.embed(data)

class Read(nn.Module):
    def __init__(self, mlp, masked=False):
        super().__init__()
        self.mlp = mlp
        self.masked = masked
    
    def forward(self, h_from, h_to, read_index):
        if read_index.size(1) == 0:
            return h_to
        message = torch_scatter.scatter(
            h_from[read_index[0]],
            read_index[1],
            dim=0,
            dim_size=h_to.size(0),
            reduce="sum"
        )
        h_to_new = torch.clone(h_to) + message
        if self.masked:
            mask = torch_scatter.scatter(
                torch.ones_like(read_index[0]),
                read_index[1],
                dim=0,
                dim_size=h_to.size(0),
                reduce="sum"
            )>0
            h_to_new[mask,:] = self.mlp(h_to_new[mask,:])
        else:
            h_to_new = self.mlp(h_to_new)
        return h_to_new

class GINEMLPConv(tgnn.MessagePassing):
    def __init__(self, pre_nn: torch.nn.Module, post_nn: torch.nn.Module,
                 eps: float = 0.,
                 train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.pre_nn = pre_nn
        self.post_nn = post_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        tgnn.inits.reset(self.pre_nn)
        tgnn.inits.reset(self.post_nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x,x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.post_nn(out)

    def message(self, x_j, edge_attr):
        return self.pre_nn(torch.cat(
            [x_j,edge_attr],dim=1
        ))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
