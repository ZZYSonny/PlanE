from plane.common_imports import *
from sage.graphs import connectivity
import networkx as nx
from . import data_process_classical

INF = 1e9

SPQR_TYPE_DICT = {
    "S": 0,
    "P": 1,
    "Q": 0,
    "R": 2
}

class Data(tgdata.Data):
    '''Definition of the data class for the PlanE'''
    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        num_g_edges = self.edge_index.size(1)
        num_g_nodes = self.x.size(0)
        num_b_nodes = self.b_batch.size(0)
        num_spqr_nodes = self.spqr_batch.size(0)
        match key:
            case 'edge_index':          return num_g_nodes

            case 'spqr_batch':          return 1
            case 'spqr_edge_index':     return num_spqr_nodes
            #case 'spqr_read_from_g':    return [[num_g_nodes], [num_spqr_nodes], [0], [0]]
            case 'spqr_read_from_e':    return [[num_spqr_nodes], [num_g_nodes], [num_g_nodes], [num_g_edges], [0], [0]]

            case 'b_batch':             return 1
            case 'b_read_from_spqr':    return [[num_spqr_nodes], [num_b_nodes]]
            case 'b_read_from_spqr_root':return [[num_spqr_nodes], [num_b_nodes]]

            case 'g_read_from_b':       return [[num_b_nodes], [num_g_nodes]]
            case 'g_read_from_spqr':    return [[num_spqr_nodes], [num_g_nodes]]

            case 'bc_edge_index':       return [[num_b_nodes], [num_g_nodes]]
            case 'cb_edge_index':       return [[num_g_nodes], [num_b_nodes]]

            case _: return 0
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if "edge_index" in key: return -1
        elif "_read_from_" in key: return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def process(data: tgdata.Data, directional_tree=True):
    def discrete(xs):
        '''Discretize the input list of values into a list of integers
            out[i]=out[j] iif xs[i]=xs[j]
        '''
        unique_xs = [x for i,x in enumerate(xs) if x not in xs[:i]]
        sorted_unique_xs = sorted(unique_xs)
        return [sorted_unique_xs.index(x) for x in xs]

    discrete_edge_attr = discrete(data.edge_attr.tolist())
    discrete_node_feature = discrete(data.x.tolist())
    discrete_edge_feature = {
        (u, v, None): discrete_edge_attr[i]
        for i in range(data.num_edges)
        for u in [data.edge_index[0,i].item()]
        for v in [data.edge_index[1,i].item()]
    }
    edge_feature_map = {
        (u, v): i
        for i in range(data.num_edges)
        for u in [data.edge_index[0,i].item()]
        for v in [data.edge_index[1,i].item()]
    }

    # [b, g]: Input Graph Node g is in the biconnected component reprsented by BC Tree Node b
    g_read_from_b = []
    # [s, g]: Input Graph Node g is in the triconnected component reprsented by SPQR Tree Node s
    g_read_from_spqr = []
    def add_graph(g: sageall.Graph):
        for cc_nodes in g.connected_components():
            cc = g.subgraph(cc_nodes)
            add_cc(cc)

    # B Node i in the BC Tree should be updated in the b_order[i] iterations
    b_order = []
    # Input Graph Node i should be updated in the c_order[i] iterations
    # If i is not a cut node, c_order[i]=-1
    c_order = [-1] * data.num_nodes
    # [b, c]: BC Tree B Node b -> the SPQR Tree C Node c
    bc_edge_index = []
    # [c, b]: SPQR Tree C Node c -> the BC Tree B Node b
    cb_edge_index = []
    
    # Number of SPQR Tree B Nodes
    b_num_nodes = 0
    # [s, b]: The triconnected component represented by SPQR Tree B Node s
    # is in the biconnected component represented by the BC Tree B Node b
    b_read_from_spqr = []
    # [s, b]: The triconnected component represented by SPQR Tree B Node s
    # is in the biconnected component represented by the BC Tree B Node b
    # and is the root of the SPQR tree
    b_read_from_spqr_root = []
    def add_cc(g: sageall.Graph):
        nonlocal b_num_nodes
        tree, bcc_tree_and_code = data_process_classical.encode_cc(g, discrete_node_feature, discrete_edge_feature)

        center = tree.center()[0]
        node_depth = {
            n: len(path)
            for n, path in tree.shortest_paths(center).items()
        }
        max_depth = max(node_depth.values())

        node_map = {}
        for node_tuple in tree.vertex_iterator():
            match node_tuple[0]:
                case 'B': 
                    # Get an id for the B node
                    b_id = b_num_nodes
                    node_map[node_tuple] = b_id
                    b_num_nodes += 1
                    # Add the computation order of the B Node b_id
                    b_order.append((max_depth - node_depth[node_tuple]) //2 )
                    # Get the SPQR Tree and the BCC Code from the classical algorithm
                    (spqr_tree, spqr_center, bcc_code) = bcc_tree_and_code[node_tuple[1]]
                    # Add the biconnected component and get the mapping from SPQR Tree Nodes to spqr id
                    b_node_map = add_bcc(spqr_tree, spqr_center, bcc_code)
                    # Construct read_froms
                    b_read_from_spqr_root.append([b_node_map[spqr_center], b_id])
                    for i in b_node_map.values():
                        b_read_from_spqr.append([i, b_id])
                    for i in node_tuple[1]:
                        g_read_from_b.append([b_id, i])
                case 'C':
                    # Add the computation order
                    c_order[node_tuple[1]] = (max_depth - node_depth[node_tuple]) //2
        
        # BC Tree edges
        for uu,vv, _ in tree.edge_iterator():
            for u,v in [(uu,vv), (vv,uu)]:
                if directional_tree and node_depth[u] < node_depth[v]:
                    continue

                if u[0] == 'B' and v[0] == 'C':
                    # b -> c
                    bc_edge_index.append([node_map[u], v[1]])
                else:
                    # c -> b
                    cb_edge_index.append([u[1], node_map[v]])

    # Number of SPQR Tree Nodes
    spqr_num_nodes = 0
    # SPQR Tree Node i should be updated in the spqr_order[i] iterations
    spqr_order = []
    # SPQR Tree Node i has type spqr_type[i]
    spqr_type = []
    # [s1,s2]: SPQR Tree Node s1 -> SPQR Tree Node s2
    spqr_edge_index = []
    # Attribute of spqr_edge_index
    spqr_edge_attr = []
    # [spqr_id, u, v, attr, i, kappa]
    # Describes the edge (u,v) in the triconnected component
    # represented by SPQR Tree Node spqr_id
    # The edge has index i in the canonical cycle of the triconnected component
    # and u has canonical code kappa
    spqr_read_from_e = []

    def add_bcc(tree: sageall.Graph, center, code):
        nonlocal spqr_num_nodes
        # Map tree node to int
        node_map = {}
        for node in tree.vertex_iterator():
            node_map[node] = spqr_num_nodes
            spqr_num_nodes += 1
            spqr_type.append(SPQR_TYPE_DICT[node[0]])
        # Find depth of each SPQR node
        node_depth = {
            node_map[node]: len(path)
            for node, path in tree.shortest_paths(center).items()
        }
        max_depth = max(node_depth.values())
        spqr_order.extend([
            max_depth - node_depth[i]
            for i in sorted(node_depth.keys())
        ])
        cycle = code.get_cycles()
        for (u,v) in reversed(list(tree.breadth_first_search(center, edges=True))):
            (spqr_cycle_u, spqr_code_u) = cycle[u]
            (spqr_cycle_v, spqr_code_v) = cycle[v]

            if not directional_tree:
            # u->v
                spqr_edge_index.append([node_map[u], node_map[v]])
                spqr_edge_attr.append(0)

            # v->u
            spqr_edge_index.append([node_map[v], node_map[u]])
            e = spqr_cycle_v[0]
            spqr_edge_attr.append(spqr_cycle_u.index(e))

        # Readout for spqr node
        for (sub_type, sub), sub_id in node_map.items():
            (spqr_cycle, spqr_code) = cycle[(sub_type, sub)]
            for i, (u,v,label) in enumerate(spqr_cycle):
                # Add u to spqr_read_from_g
                #spqr_read_from_g.append([u, sub_id, i, spqr_code[i]])

                # Add edge to spqr_read_from_e
                if label is None:
                    spqr_read_from_e.append([sub_id, u, v, edge_feature_map[(u,v)], i, spqr_code[i]])
                else:
                    spqr_read_from_e.append([sub_id, u, v, -INF, i, spqr_code[i]])
                
                if sub_type == 'P':
                    #spqr_read_from_g.append([v, sub_id, i, spqr_code[i]])
                    if label is None:
                        spqr_read_from_e.append([sub_id, v, u, edge_feature_map[(v,u)], i, spqr_code[i]])
                    else:
                        spqr_read_from_e.append([sub_id, v, u, -INF, i, spqr_code[i]])

            for i in sub.vertex_iterator():
                g_read_from_spqr.append([sub_id, i])
        
        return node_map
        
    add_graph(sageall.Graph(data.edge_index.T.tolist()))
    return Data(
        x = data.x,
        y = data.y,
        edge_index = data.edge_index,
        edge_attr = data.edge_attr,
        
        spqr_type = torch.tensor(spqr_type, dtype=torch.long),
        spqr_batch = torch.zeros(spqr_num_nodes, dtype=torch.long),
        spqr_order = torch.tensor(spqr_order, dtype=torch.long),
        spqr_edge_index = torch.tensor(spqr_edge_index, dtype=torch.long).view(-1,2).mT,
        spqr_edge_attr = torch.tensor(spqr_edge_attr, dtype=torch.long),
        spqr_read_from_e = torch.tensor(spqr_read_from_e, dtype=torch.long).view(-1,6).mT,
        
        b_batch = torch.zeros(b_num_nodes, dtype=torch.long),
        b_read_from_spqr = torch.tensor(b_read_from_spqr, dtype=torch.long).view(-1,2).mT,
        b_read_from_spqr_root = torch.tensor(b_read_from_spqr_root, dtype=torch.long).view(-1,2).mT,

        g_read_from_b = torch.tensor(g_read_from_b, dtype=torch.long).view(-1,2).mT,
        g_read_from_spqr = torch.tensor(g_read_from_spqr, dtype=torch.long).view(-1,2).mT,

        b_order = torch.tensor(b_order, dtype=torch.long),
        c_order = torch.tensor(c_order, dtype=torch.long),
        bc_edge_index = torch.tensor(bc_edge_index, dtype=torch.long).view(-1,2).mT,
        cb_edge_index = torch.tensor(cb_edge_index, dtype=torch.long).view(-1,2).mT,
    )


def add_zero_node_attr(data: tgdata.Data):
    data.x = torch.zeros((data.num_nodes,), dtype=torch.long)
    return data

def add_zero_edge_attr(data: tgdata.Data):
    data.edge_attr = torch.zeros((data.num_edges,), dtype=torch.long)
    return data

def node_cluster_coefficient_graph(data: tgdata.Data):
    g_nx = tgutils.to_networkx(data)
    coeff = nx.algorithms.cluster.clustering(g_nx)
    return tgdata.Data(
        x = torch.zeros((data.num_nodes,), dtype=torch.long),
        edge_index = data.edge_index,
        edge_attr = torch.zeros((data.num_edges,), dtype=torch.long),
        y = torch.tensor([
            coeff[i]
            for i in range(data.num_nodes)
        ], dtype=torch.float)
    )

def graph_cluster_coefficient_graph(data: tgdata.Data):
    g_nx = tgutils.to_networkx(data)
    coeff = nx.algorithms.cluster.transitivity(g_nx)
    return tgdata.Data(
        x = torch.zeros((data.num_nodes,), dtype=torch.long),
        edge_index = data.edge_index,
        edge_attr = torch.zeros((data.num_edges,), dtype=torch.long),
        y = torch.tensor(coeff, dtype=torch.float)
    )