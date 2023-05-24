from utils.common_imports_experiment import *
from utils.processing.datasets import QM9Dataset
import gzip
import json

def transform(data):
    data1 = data_process.graph_cluster_coefficient_graph(data)
    if 0.06 <= data1.y <= 0.16:
        return data_process.process(data1)
    else:
        return None

root = ".dataset/QM9CCTMP"
train = QM9Dataset(root=root, split="train", pre_transform=transform)
valid = QM9Dataset(root=root, split="valid", pre_transform=transform)
test  = QM9Dataset(root=root, split="test", pre_transform=transform)

for name, dataset in [("train", train), ("valid", valid), ("test", test)]:
    loader = tgdata.DataLoader(dataset, batch_size=1, shuffle=False)
    data_json = []
    for data in loader:
        data_json.append({
            "x": data.x.tolist(),
            "edge_index": data.edge_index.tolist(),
            "edge_attr": data.edge_attr.tolist(),
            "y": data.y.tolist()
        })

    with gzip.open(f"{root}/{name}.pkl.gz", "wt") as f:
        json.dump(data_json, f)
