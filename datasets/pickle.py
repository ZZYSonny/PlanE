import pickle
from tqdm import tqdm
import torch
import torch_geometric.data as tgdata

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

def LargeGraphDataset(root, size, transform=None, pre_transform=None):
    return PickleDataset(f".dataset_src/Large_{size}.pkl", f"{root}/{size}", transform, pre_transform)