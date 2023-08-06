from utils.common_imports_experiment import *
from utils.processing.datasets import P3RDataset


arg = wandb_init()
if arg is None:
    print("Using Arg Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_hidden", type=int, default=48)
    parser.add_argument("--dim_plane_pe", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--flags_layer", type=str, default="plane")
    parser.add_argument("--flags_plane_agg", type=str, default="n_t_b_gr_cr")
    parser.add_argument("--flags_mlp_factor", type=int, default=2)
    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--total_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--total_split", type=int, default=10)
    parser.add_argument("--cur_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    arg = parser.parse_args()


def get_model():
    model_config = models.ModelConfig(
        dim=arg.dim_hidden,
        dim_plane_pe=arg.dim_plane_pe,
        num_layers=arg.num_layers,
        dim_node_feature=[1],
        dim_edge_feature=None,
        dim_output=9,
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
    root = ".dataset/P3R"

    transform = tgtrans.Compose(
        [
            data_process.add_zero_edge_attr,
            data_process.process,
        ]
    )
    dataset = P3RDataset(root, pre_transform=transform)

    i = arg.cur_split
    SPLITS = arg.total_split

    n = len(dataset) // SPLITS
    test_mask = torch.zeros(len(dataset), dtype=torch.bool)

    test_mask[i * n : (i + 1) * n] = 1  # Now set the masks

    # Now load the datasets
    test_dataset = dataset[test_mask]
    train_dataset = dataset[~test_mask]

    n = len(train_dataset) // SPLITS
    val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    val_mask[i * n : (i + 1) * n] = 1
    val_dataset = train_dataset[val_mask]
    train_dataset = train_dataset[~val_mask]

    return (train_dataset, val_dataset, test_dataset)


def main():
    exec_config = ExecutionConfig(
        num_epoch=arg.total_epoch,
        batch_size=arg.batch_size,
        seed=arg.seed,
        log_loss_freq=1,
    )

    Trainer(
        get_model=get_model,
        get_optimizer=get_optimizer,
        get_scheduler=get_scheduler,
        get_dataset=get_dataset,
        find_loss=find_loss,
        find_metric=find_metric,
        exec_config=exec_config,
    ).run()

if __name__ == "__main__":
    main()
