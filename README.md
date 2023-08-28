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
To run an experiment, you will first need to prepare the dataset. Simply run the following command to download and prepare all datasets used in the paper.

```bash
python3 -m preprocess.prepare
```

We use [WanDB](https://wandb.ai/) to manage hyper-parameter searching and log experiement results. You can use the searching grid we used in the paper by creating a sweep from the [experiments/config](experiments/config) folder. For example, `wandb sweep experiments/config/express_synth/exp/plane.yaml` will create a sweep to tune PlanE on the EXP dataset. After creating a sweep, you can find the command to launch the sweep from the command output. They are usually in the form of `wandb agent <username>/<project>/<sweep_id>`. 

Once the experiment is launched, you can find the result in the WanDB dashboard. The training/validation/test metric is logged as `train`, `valid`, `test`. Note we report negative MAE for regression tasks as this aligns with other datasets where the goal is to maximize some metric. 

## Synthetic Dataset: QM9CC
We designed the QM9CC dataset to evaluate the model's ability to detect structural graph signals \emph{without} an explicit reference to the target structure. The dataset is generated from a subset of graphs from the QM9 dataset and the goal is to predict the clustering coefficient of the graph.

You may want to use the QM9CC dataset for your models. Simply copy [datasets/qm9cc_portable.py](datasets/qm9cc_portable.py) and [.dataset_src](.dataset_src) folder to your project. 

## Synthetic Dataset: P3R
The P3R dataset is made from 9 planar 3-regular graphs of size 10. The training, validation and test set are permutations of the 9 graphs. The goal is to predict a number from 0 to 8, which is the index of the graph from the 9 P3R graphs. 