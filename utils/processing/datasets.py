from utils.common_imports import *
import pickle
import os.path as osp
import gzip
import json

class PickleDataset(tgdata.InMemoryDataset):
    def __init__(self, src, root, transform=None, pre_transform=None):
        self.src = src
        super(PickleDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        data_list_old = pickle.load(open(self.src, 'rb'))
        data_list_new = []
        for data in tqdm(data_list_old):
            new_data = self.pre_transform(data)
            data_list_new.append(new_data)
        data, slices = self.collate(data_list_new)
        torch.save((data, slices), self.processed_paths[0])

def EXPDataset(root, transform=None, pre_transform=None):
    return PickleDataset(".dataset_src/EXP.pkl", root, transform, pre_transform)

def P3RDataset(root, transform=None, pre_transform=None):
    return PickleDataset(".dataset_src/P3R.pkl", root, transform, pre_transform)

def map_qm9_to_pyg(json_file, make_undirected=True, remove_dup=False):
    # We're making the graph undirected just like the original repo.
    # Note: make_undirected makes duplicate edges, so we need to preserve edge types.
    # Note: The original repo also add self-loops. We don't need that given how we see hops.
    edge_index = np.array([[g[0], g[2]] for g in json_file["graph"]]).T  # Edge Index
    edge_attributes = np.array(
        [g[1] - 1 for g in json_file["graph"]]
    )  # Edge type (-1 to put in [0, 3] range)
    if (
        make_undirected
    ):  # This will invariably cost us edge types because we reduce duplicates
        edge_index_reverse = edge_index[[1, 0], :]
        # Concat and remove duplicates
        if remove_dup:
            edge_index = torch.LongTensor(
                np.unique(
                    np.concatenate([edge_index, edge_index_reverse], axis=1), axis=1
                )
            )
        else:
            edge_index = torch.LongTensor(
                np.concatenate([edge_index, edge_index_reverse], axis=1)
            )
            edge_attributes = torch.LongTensor(
                np.concatenate([edge_attributes, np.copy(edge_attributes)], axis=0)
            )
    x = torch.FloatTensor(np.array(json_file["node_features"]))
    y = torch.FloatTensor(np.array(json_file["targets"]).T)
    return tgdata.Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)


class QM9Dataset(tgdata.InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        self.json_gzip_path = f".dataset_src/QM9_{split}.jsonl.gz"
        new_root = osp.join(root, split)
        super(QM9Dataset, self).__init__(new_root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        data_list_new = []

        with gzip.open(self.json_gzip_path, "rb") as f:
            data = f.read().decode("utf-8")
            for line in tqdm(data.splitlines()):
                graph_dict = json.loads(line)
                graph_torch = map_qm9_to_pyg(graph_dict)
                graph_tran = self.pre_transform(graph_torch)
                if self.pre_filter is not None:
                    if not self.pre_filter(graph_tran):
                        continue
                if graph_tran is not None:
                    data_list_new.append(graph_tran)

        data, slices = self.collate(data_list_new)
        torch.save((data, slices), self.processed_paths[0])
