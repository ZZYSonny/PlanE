# PlanE: Representation Learning over Planar Graphs
The repository contains the code for the paper "Representation Learning over Planar Graphs". 

## Install
We provide all the dependencies in a conda environment. You can create the environment by running
```bash
conda env create -f environment.yml
```

Then activate the environment by running
```bash
conda activate plane
```

You may also use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to speed up environment installation.

## Run the models
We use [WanDB](https://wandb.ai/) to track the experiments. You can create a free account and log in to track your experiments.

To run an experiment with WanDB, you first need to create a sweep. 
```bash
# EXP Dataset
wandb sweep config/exp.yaml
# QM9
wandb sweep config/qm9.yaml
# ZINC Subset (12k)
wandb sweep config/zinc12k.yaml
# ZINC Subset (12k) without edge feature
wandb sweep config/zinc12knoe.yaml
# ZINC Full dataset
wandb sweep config/zincfull.yaml
# OGB-MolHIV
wandb sweep config/ogb-molhiv.yaml
## CC PlanE
wandb sweep config/cc-plane.yaml
## CC GCN/GIN
wandb sweep config/cc-gnn.yaml
```

You can find the command to launch the experiments from the command output. They are usually in the form of `wandb agent <username>/<project>/<sweep_id>`. 

Once the experiment is launched, you can find the result in the WanDB dashboard. The training/validation/test metric is logged as `train`, `valid`, `test`. Note we report negative MAE for regression tasks as this aligns with other datasets where the goal is to maximize some metric. 

## Synthetic Dataset: QM9CC
We designed the QM9CC dataset to evaluate the model's ability to detect structural graph signals \emph{without} an explicit reference to the target structure. The dataset is generated from a subset of graphs from the QM9 dataset and the goal is to predict the clustering coefficient of the graph.

You may want to use the QM9CC dataset for your models. Simply copy [experiments/cc/qm9cc_portable.py](experiments/cc/qm9cc_portable.py) and [.dataset_src](.dataset_src) folder to your project. 