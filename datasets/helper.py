import torch_geometric.datasets as tgsets
import torch

def get_dataset(name: str, fn_final_transform=None, split: str = "", tag:str="v0"):
    root = f".dataset/{tag}/{name}"
    pre_transform = fn_final_transform or (lambda x:x)

    # Dataset where data is splitted using the split argument
    fn_dataset = None
    match name:
        case "QM9" | "QM9NoE":
            from datasets.qm9 import QM9Dataset
            fn_dataset = lambda split: QM9Dataset(root=root, split=split, pre_transform=pre_transform)
        case "QM9CC":
            from datasets.qm9 import QM9Dataset
            from datasets.qm9cc import transform_wrapper
            fn_dataset = lambda split: QM9Dataset(root=root, split=split, pre_transform=lambda g: transform_wrapper(g, fn_final_transform))
        case "ZINC12k" | "ZINC12kNoE":
            from torch_geometric.datasets import ZINC
            fn_dataset = lambda split: ZINC(root=root, subset=True, split=split, pre_transform=pre_transform)
        case "ZINCFull":
            from torch_geometric.datasets import ZINC
            fn_dataset = lambda split: ZINC(root=root, subset=False, split=split, pre_transform=pre_transform)
        case "ogbg_molhiv":
            from datasets.ogb import ogbg_dataset_wrapper
            fn_dataset = lambda split: ogbg_dataset_wrapper("ogbg_molhiv", root=root, split=split, pre_transform=pre_transform)            

    if fn_dataset is not None:
        train = fn_dataset("train")
        valid = fn_dataset("val")
        test  = fn_dataset("test")

        if name=="QM9" and split != "None":
            cur_split = int(split)
            train.data.y = train.data.y[:, cur_split]
            valid.data.y = valid.data.y[:, cur_split]
            test.data.y = test.data.y[:, cur_split]
        
        return train, valid, test

    # Dataset where data is splitted using ncross validation
    match name:
        case "P3R":
            from datasets.pickle import P3RDataset
            fn_dataset = lambda: P3RDataset(root=root, pre_transform=pre_transform)
        case "EXP":
            from datasets.pickle import EXPDataset
            fn_dataset = lambda: EXPDataset(root=root, pre_transform=pre_transform)
        
    if fn_dataset is not None:
        dataset = fn_dataset()

        i = int(split.split("_")[0])
        total_split = int(split.split("_")[1])
        
        n = len(dataset) // total_split
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)

        test_mask[i * n : (i + 1) * n] = 1  # Now set the masks

        # Now load the datasets
        test = dataset[test_mask]
        train = dataset[~test_mask]

        n = len(train) // total_split
        val_mask = torch.zeros(len(train), dtype=torch.bool)
        val_mask[i * n : (i + 1) * n] = 1
        valid = train[val_mask]
        train = train[~val_mask]

        return train, valid, test
    
    if name == "Large_Tiger_Alaska":
        return dataset, None, None
    
    raise Exception(f"Unknown Dataset {name}")


def get_dataset_info(name):
    match name:
        case "QM9": return {
            "dim_node_feature": "lin",
            "dim_edge_feature": 4,
            "dim_output": 1
        }
        case "QM9NoE": return {
            "dim_node_feature": "lin",
            "dim_edge_feature": "None",
            "dim_output": 1
        }
        case "QM9CC": return {
            "dim_node_feature": 1,
            "dim_edge_feature": "None",
            "dim_output": 1
        }
        case "ZINC12k" | "ZINCFull": return {
            "dim_node_feature": [32],
            "dim_edge_feature": 4,
            "dim_output": 1
        }
        case "ZINC12kNoE": return {
            "dim_node_feature": [32],
            "dim_edge_feature": "None",
            "dim_output": 1
        }
        case "ogbg_molhiv": return {
            "dim_node_feature": "ogb_atom_node",
            "dim_edge_feature": "ogb_atom_edge",
            "dim_output": 1
        }
        case "P3R": return {
            "dim_node_feature": [1],
            "dim_edge_feature": "None",
            "dim_output": 9
        }
        case "EXP": return {
            "dim_node_feature": [2],
            "dim_edge_feature": "None",
            "dim_output": 1
        }
        case "Large_Tiger_Alaska": return {
            "dim_node_feature": [1],
            "dim_edge_feature": "None",
            "dim_output": 2
        }
        case _:
            raise Exception(f"Unknown Dataset {name}")