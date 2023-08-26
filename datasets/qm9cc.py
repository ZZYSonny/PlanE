import torch_geometric.data as tgdata
import torch_geometric.utils as tgutils
import torch
import networkx as nx

def graph_cluster_coefficient_graph(data: tgdata.Data):
    g_nx = tgutils.to_networkx(data)
    coeff = nx.algorithms.cluster.transitivity(g_nx)
    return tgdata.Data(
        x = torch.zeros((data.num_nodes,), dtype=torch.long),
        edge_index = data.edge_index,
        edge_attr = torch.zeros((data.num_edges,), dtype=torch.long),
        y = torch.tensor(coeff, dtype=torch.float)
    )

def transform_wrapper(data: tgdata.Data, fn_transform):
    data1 = graph_cluster_coefficient_graph(data)
    if 0.06 <= data1.y <= 0.16:
        return fn_transform(data1)
    else:
        return None
