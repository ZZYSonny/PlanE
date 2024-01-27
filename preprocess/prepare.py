from datasets.helper import get_dataset
from preprocess.data_process import process


for dataset in [
    "QM9",
    "QM9NoE",
    "QM9CC",
    "ZINC12k",
    "ZINCFull",
    "ZINC12kNoE",
    "ogbg_molhiv",
    "P3R",
    "EXP",
]:
    get_dataset(name=dataset, fn_final_transform=process)
