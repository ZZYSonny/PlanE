import argparse
from datasets.pickle import LargeGraphDataset
from experiments.trainer import *
from plane import models
from plane.common_imports import *
from preprocess import data_process
import time


wandb_init()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Tiger_Alaska_10k")
parser.add_argument("--total_epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--dim_hidden", type=int, default=32)
parser.add_argument("--dim_plane_pe", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--flags_layer", type=str, default="plane")
parser.add_argument("--flags_plane_agg", type=str, default="n_t_b_gr_cr")
parser.add_argument("--flags_mlp_factor", type=int, default=2)
parser.add_argument("--lr_start", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
arg = parser.parse_args()


def get_model():
    model_config = models.ModelConfig(
        dim=arg.dim_hidden,
        dim_plane_pe=arg.dim_plane_pe,
        num_layers=arg.num_layers,
        dim_node_feature=[1],
        dim_edge_feature="None",
        dim_output=2,
        flags_layer=arg.flags_layer,
        flags_plane_agg=arg.flags_plane_agg,
        flags_mlp_factor=arg.flags_mlp_factor,
        drop_agg=0,
        drop_com=0,
        drop_enc=0,
        drop_out=0,
        drop_rec=0,
        drop_edg=0,
    )
    return models.ModelGraph(model_config)


def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=arg.lr_start)


def get_scheduler(optimizer):
    return None


loss_func = nn.CrossEntropyLoss()


def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return loss_func(model_out, dataset_y)


def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    model_y = model_out.argmax(dim=-1)
    return (model_y == dataset_y).float().mean().item()


def get_dataset():
    if arg.flags_layer == "plane":
        transform = tgtrans.Compose(
            [
                data_process.add_zero_edge_attr,
                data_process.process,
            ]
        )
    else:
        transform = tgtrans.Compose(
            [
                data_process.add_zero_edge_attr,
            ]
        )
    dataset = LargeGraphDataset(
        f".dataset/v0/{arg.flags_layer}",
        arg.dataset_name,
        pre_transform=transform,
    )
    return [dataset[0]] * arg.batch_size


# Loading Dataset
time_start = time.time()
dataset = get_dataset()
time_end = time.time()
print(f"Dataset Loaded in {time_end - time_start}s")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
optimizer = get_optimizer(model)
loader = tgloader.DataLoader(dataset, batch_size=arg.batch_size)


def train_one_epoch():
    model.train()
    for batch in loader:
        model_out = model(batch.to(device))
        optimizer.zero_grad()
        loss = find_loss(model_out, batch.y)
        loss.backward()
        optimizer.step()


def evaluate_model():
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.to(device))
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        metric = find_metric(y_pred, y_true)
        return metric


print("Warming Up Training")
train_one_epoch()
time_logs = []
for epoch in range(arg.total_epoch):
    time_start = time.time()
    train_one_epoch()
    time_end = time.time()
    time_logs.append(time_end - time_start)
    print(f"    Epoch {epoch} Time {time_logs[-1]}s")
print(f"Training Time Per Epoch {np.mean(time_logs)} +- {np.std(time_logs)}s")

print("Warming Up Validation")
evaluate_model()
time_logs = []
for epoch in range(arg.total_epoch):
    time_start = time.time()
    evaluate_model()
    time_end = time.time()
    time_logs.append(time_end - time_start)
    print(f"    Epoch {epoch} Time {time_logs[-1]}s")
print(
    f"Validation Time Per Epoch {np.mean(time_logs)} +- {np.std(time_logs)}s"
)
