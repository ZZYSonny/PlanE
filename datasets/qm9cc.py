import networkx as nx
import torch
from torch_geometric import data as tgdata, utils as tgutils


def graph_cluster_coefficient_graph(data: tgdata.Data):
    g_nx = tgutils.to_networkx(data)
    coeff = nx.algorithms.cluster.transitivity(g_nx)
    return tgdata.Data(
        x=torch.zeros((data.num_nodes,), dtype=torch.long),
        edge_index=data.edge_index,
        edge_attr=torch.zeros((data.num_edges,), dtype=torch.long),
        y=torch.tensor(coeff, dtype=torch.float),
    )


def transform_wrapper(data: tgdata.Data, fn_transform):
    data1 = graph_cluster_coefficient_graph(data)
    if 0.06 <= data1.y <= 0.16:
        return fn_transform(data1)
    else:
        return None
