from utils.common_imports import *
from .tools import *

class TriEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.h_virtual = nn.Parameter(torch.randn((config.dim)))
        if config.dim_edge_feature is None:
            self.h_edge = nn.Parameter(torch.randn((config.dim)))
        self.pe = tgnn.PositionalEncoding(config.dim_plane_pe, base_freq=1/64)
        self.lin_node = MLP(2*config.dim+2*config.dim_plane_pe, config.dim, drop=config.drop_enc, factor=config.flags_mlp_factor)
        self.mlp_out = nn.ModuleList([
            MLP(config.dim, config.dim, drop=config.drop_enc, factor=config.flags_mlp_factor, norm=None)
            for _ in range(3)
        ])
        self.bn_out = nn.BatchNorm1d(config.dim)

    def forward(self, data, h_g, h_e):
        # Number of triconnected components
        n_spqr = data.spqr_batch.size(0)

        # Edges in all triconnected components, for each edge, we have
        #   id_spqr: which triconnected component the edge belongs to
        #   id_u: the source node of the edge
        #   id_v: the target node of the edge
        #   id_e: If <0, the edge is a virtual edge
        #         if >=0,the edge is a real edge 
        #           and id_e is the index of the edge feature
        #   code1: the index of the edge in the canonical walk
        #   code2: \kappa[code1]
        id_spqr, id_u, id_v, id_e, code1, code2 = data.spqr_read_from_e
        # Initialize the edge feature
        h_cycle_edge = torch.zeros((id_e.size(0), self.config.dim), device=h_g.device)
        # For virtual edges, we use the learnable virtual edge feature
        virtual_mask = (id_e < 0)
        h_cycle_edge[virtual_mask] = self.h_virtual
        if h_e is None:
            # If no edge feature is provided, we use the learnable edge feature
            # As we want to distinguish virtual edges and real edges
            h_cycle_edge[~virtual_mask] = self.h_edge
        else:
            # Read the edge feature
            h_cycle_edge[~virtual_mask] = h_e[id_e[~virtual_mask]]

        # Construct the tuple and perform the first MLP (to mix the channels)
        pre_agg = self.lin_node(torch.cat([
            h_g[id_u],
            h_cycle_edge,
            self.pe(code1),
            self.pe(code2)
        ],dim=1))

        # Aggregate in terms of triconnected components
        pre_mlp = torch_scatter.scatter(
            pre_agg,
            id_spqr,
            dim=0,
            dim_size=data.spqr_batch.size(0),
            reduce="add"
        )

        # Apply a different MLP for each type of triconnected components
        out = torch.zeros((n_spqr, self.config.dim), device=data.x.device)
        for i in range(3):
            mask = data.spqr_type==i
            out[mask] = self.mlp_out[i](pre_mlp[mask])

        # Perform batch normalization on all triconnected components representations
        out = self.bn_out(out)

        return out


class BiRecEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.flags_plane_gine_type == "complete":
            self.pe = tgnn.PositionalEncoding(config.dim_plane_pe, base_freq=1/64)
            self.update = GINEMLPConv(
                MLP(config.dim + config.dim_plane_pe, config.dim, drop=config.drop_edg, factor=config.flags_mlp_factor),
                nn.Identity(),
                train_eps=True
            )
        elif config.flags_plane_gine_type == "incomplete":
            self.pe = tgnn.PositionalEncoding(config.dim, base_freq=1/64)
            self.update = tgnn.GINEConv(nn.Identity())
        else:
            raise NotImplementedError
        self.mlp_after_update = MLP(config.dim, config.dim, drop=config.drop_rec, factor=config.flags_mlp_factor, norm=None)
        self.read = tgnn.GINConv(MLP(config.dim, config.dim, drop=config.drop_enc, factor=config.flags_mlp_factor))

    def forward(self, data, h_spqr):
        h_spqr = h_spqr.clone()
        # Embed \theta into positional encoding vectors
        h_spqr_edge = self.pe(data.spqr_edge_attr)
        for cur_order in range(max(data.spqr_order)+1):
            # At the first iteration, we update node with no children
            # At the second iteration, we update node whose children are updated at the first iteration etc
            # The node is recognized by data.spqr_order
            node_mask = data.spqr_order==cur_order
            # Edges connecting the node to its children
            edge_mask = node_mask[data.spqr_edge_index[1]]
            h_new_spqr = self.update(
                h_spqr.clone(),
                edge_index=data.spqr_edge_index[:, edge_mask],
                edge_attr=h_spqr_edge[edge_mask]
            )[node_mask]
            # Write back
            h_new_spqr = self.mlp_after_update(h_new_spqr)
            h_spqr[node_mask] = h_new_spqr
        
        # Obtain a representation for each biconnected component
        # By reading from the canonical center of the SPQR tree
        h_b = self.read(
            (h_spqr,torch.zeros((data.b_batch.size(0), self.config.dim), device=data.x.device)),
            data.b_read_from_spqr_root,
        )
        return h_b

class CnearBEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.mp = tgnn.GINConv(MLP(config.dim, config.dim, drop=config.drop_enc, factor=config.flags_mlp_factor))
    
    def forward(self, data, h_g, h_b):
        # A cut encoder that simply perform a few rounds of message passing
        return self.mp(
            (h_b, h_g),
            data.bc_edge_index
        )

class CRecSubEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.update = tgnn.GINConv(nn.Identity())
        self.mlp_after_update = MLP(config.dim, config.dim, drop=config.drop_rec, factor=config.flags_mlp_factor, norm=None)
        self.read_b = tgnn.GINConv(nn.Identity())
        self.read_c = tgnn.GINConv(nn.Identity())
        self.mlp_out = MLP(config.dim, config.dim, drop=config.drop_enc, factor=config.flags_mlp_factor)

    def forward(self, data, h_g, h_b):
        max_order = max(
            data.b_order.max().item(),
            data.c_order.max().item()
        )
        h_g = h_g.clone()
        h_b = h_b.clone()
        for cur_order in range(max_order+1):
            # Mostly the same logic as BiRecEncoder
            # Except that we have two types of nodes
            # We first deal with biconnected components (B nodes) and then cut nodes (C nodes)
            b_node_mask = data.b_order==cur_order
            cb_edge_mask = b_node_mask[data.cb_edge_index[1]]
            h_b = self.update(
                (h_g, h_b),
                edge_index=data.cb_edge_index[:, cb_edge_mask],
            )
            h_b[b_node_mask] = self.mlp_after_update(h_b[b_node_mask])

            c_node_mask = data.c_order==cur_order
            bc_edge_mask = c_node_mask[data.bc_edge_index[1]]
            h_g = self.update(
                (h_b, h_g),
                edge_index=data.bc_edge_index[:, bc_edge_mask],
            )
            h_g[c_node_mask] = self.mlp_after_update(h_g[c_node_mask])

        cut_mask = data.c_order>=0
        h_g[~cut_mask] = 0      
        return h_g




class PlaneLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.flag_aggr_neigh = 'n' in config.plane_terms
        self.flag_aggr_spqr = 't' in config.plane_terms
        self.flag_aggr_b = 'b' in config.plane_terms
        self.flag_aggr_planar_readout = 'cr' in config.plane_terms or 'cnb' in config.plane_terms
        self.flag_aggr_global_readout = 'gr' in config.plane_terms
        self.flag_compute_planar_readout = self.flag_aggr_planar_readout
        self.flag_compute_b = self.flag_aggr_b or self.flag_compute_planar_readout
        self.flag_compute_spqr = self.flag_aggr_spqr or self.flag_compute_b

        self.embed_e = Embedding(config.dim_edge_feature, config.dim)

        if self.flag_compute_spqr:
            self.encoder_spqr = TriEncoder(config)
        
        if self.flag_compute_b:
            if 'cnb' in config.plane_terms:
                self.encoder_b = CnearBEncoder(config)
            else:
                self.encoder_b = BiRecEncoder(config)
        
        if self.flag_compute_planar_readout:
            self.encoder_pr = CRecSubEncoder(config)

        if self.flag_aggr_global_readout:
            self.encoder_gr = tgnn.DeepSetsAggregation(nn.Identity(), MLP(config.dim, config.dim, drop=config.drop_agg, factor=config.flags_mlp_factor))

        if self.flag_aggr_neigh:
            if config.dim_edge_feature is not None:
                if config.flags_plane_gine_type == "complete":
                    self.aggr_neigh = GINEMLPConv(
                        MLP(config.dim + config.dim, config.dim, factor=-1 if config.flags_mlp_factor==-1 else config.flags_mlp_factor//2, drop=config.drop_edg),
                        MLP(config.dim, config.dim, factor=config.flags_mlp_factor, drop=config.drop_agg),
                        train_eps=True
                    )
                elif config.flags_plane_gine_type == "incomplete":
                    self.aggr_neigh = tgnn.GINEConv(MLP(config.dim, config.dim, drop=config.drop_agg, factor=config.flags_mlp_factor), train_eps=True)
                else:
                    raise NotImplementedError
            else:
                self.aggr_neigh = tgnn.GINConv(MLP(config.dim, config.dim, drop=config.drop_agg, factor=config.flags_mlp_factor), train_eps=True)
        
        if self.flag_aggr_spqr:
            self.aggr_spqr = tgnn.GINConv(MLP(config.dim, config.dim, drop=config.drop_agg, factor=config.flags_mlp_factor), train_eps=True)

        if self.flag_aggr_b:
            self.aggr_b = tgnn.GINConv(MLP(config.dim, config.dim, drop=config.drop_agg, factor=config.flags_mlp_factor), train_eps=True)
        
        self.mlp_out = nn.Sequential(
            nn.LazyLinear(config.dim),
            nn.BatchNorm1d(config.dim),
            nn.ReLU(),
            nn.Dropout(config.drop_com),
        )

    def forward(self, data, h_g):
        h_e = self.embed_e(data.edge_attr)
        aggs = []

        if self.flag_aggr_neigh:
            # Aggregate from neighbors
            if h_e is not None:
                # If edge features are present
                aggs.append(self.aggr_neigh(
                    (h_g, h_g),
                    edge_index=data.edge_index,
                    edge_attr=h_e
                ))
            else:
                # No edge features
                aggs.append(self.aggr_neigh(
                    h_g,
                    edge_index=data.edge_index,
                ))
        
        if self.flag_aggr_global_readout:
            # Compute the global readout for each graph
            # Aggregate from the global readout
            h_gr = self.encoder_gr(
                h_g,
                data.batch,
                dim_size = data.num_graphs
            )
            aggs.append(h_gr[data.batch])

        if self.flag_compute_spqr:
            # Compute (SPQR) components features
            h_spqr = self.encoder_spqr(data, h_g, h_e)
            if self.flag_aggr_spqr:
                # Aggregate from SPQR components
                aggs.append(self.aggr_spqr(
                    (h_spqr,h_g),
                    edge_index=data.g_read_from_spqr,
                ))

            if self.flag_compute_b:
                # Compute biconnected component features
                h_b = self.encoder_b(data, h_spqr)
                if self.flag_aggr_b:
                    # Aggregate from biconnected components
                    aggs.append(self.aggr_b(
                        (h_b,h_g),
                        edge_index=data.g_read_from_b,
                    ))
            
                if self.flag_compute_planar_readout:
                    # Compute cut readout
                    h_pr = self.encoder_pr(data, h_g, h_b)
                    if self.flag_aggr_planar_readout:
                        # Aggregate from cut readout
                        aggs.append(h_pr)

        h_new_g = self.mlp_out(torch.cat(aggs, dim=1))
        
        return h_new_g
