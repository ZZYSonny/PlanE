from datasets.helper import get_dataset
from preprocess.data_process import process


for dataset in ["ZINC12k"]:
    get_dataset(dataset,process)