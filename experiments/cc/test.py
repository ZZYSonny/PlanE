from utils.common_imports_experiment import *
from utils.processing.datasets import QM9Dataset

arg = wandb_init()
if arg is None:
    print("Using Arg Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=32)
    parser.add_argument('--dim_plane_pe', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--flags_layer', type=str, default="plane")
    parser.add_argument('--flags_plane_agg', type=str, default="n_t_b_gr_cr")
    parser.add_argument('--lr_start', type=float, default=1e-4)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    arg = parser.parse_args()

def get_model():
    model_config = models.ModelConfig(
        dim = arg.dim_hidden,
        dim_plane_pe = arg.dim_plane_pe,
        num_layers = arg.num_layers,
        dim_node_feature = 1,
        dim_edge_feature = None,
        dim_output = 1,
        flags_layer=arg.flags_layer,
        flags_plane_agg=arg.flags_plane_agg,
        drop_agg=0,
        drop_com=0,
        drop_enc=0,
        drop_out=0,
        drop_rec=0,
        drop_edg=0,
        act_out="Sigmoid"
    )
    return models.ModelGraph(model_config)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=arg.lr_start)

def get_scheduler(optimizer):
    return None

loss_func = nn.MSELoss()
def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return loss_func(model_out.flatten(), dataset_y.flatten())

metric_func = nn.L1Loss()
def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return -metric_func(model_out.flatten(), dataset_y.flatten())

def get_dataset():
    root = ".dataset/QM9CC"
    def transform(data):
        data1 = data_process.graph_cluster_coefficient_graph(data)
        if 0.06 <= data1.y <= 0.16:
            return data_process.process(data1)
        else:
            return None

    train = QM9Dataset(root=root, split="train", pre_transform=transform)
    valid = QM9Dataset(root=root, split="valid", pre_transform=transform)
    test  = QM9Dataset(root=root, split="test", pre_transform=transform)

    train.data.y = (train.data.y - 0.06) * 10
    valid.data.y = (valid.data.y - 0.06) * 10
    test.data.y  = (test.data.y - 0.06) * 10
    return (train, valid, test)


def main():
    exec_config = ExecutionConfig(
        num_epoch = arg.total_epoch,
        batch_size = arg.batch_size,
        seed = arg.seed,
        save_cp_freq=1,
        save_best_state="test"
        #log_loss_freq=40
    )
    
    Trainer(
        get_model = get_model,
        get_optimizer = get_optimizer,
        get_scheduler= get_scheduler,
        get_dataset = get_dataset,
        find_loss = find_loss,
        find_metric = find_metric,
        exec_config= exec_config
    ).run()

main()