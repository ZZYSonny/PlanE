from utils.common_imports_experiment import *
from utils.processing.datasets import QM9Dataset

CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]

arg = wandb_init()
if arg is None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=128)
    parser.add_argument('--dim_plane_pe', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--flags_layer', type=str, default="plane")
    parser.add_argument('--flags_plane_agg', type=str, default="n_t_b_gr_cr")
    parser.add_argument('--flags_mlp_factor', type=int, default=2)
    parser.add_argument('--lr_start', type=float, default=5e-4)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_step_size', type=int, default=20)
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=int, default=0)
    arg = parser.parse_args()
target = arg.target

def get_model():
    model_config = models.ModelConfig(
        dim = arg.dim_hidden,
        dim_plane_pe = arg.dim_plane_pe,
        num_layers = arg.num_layers,
        dim_node_feature = "lin",
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
    )
    return models.ModelGraph(model_config)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=arg.lr_start)

def get_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=arg.lr_step_size, 
        gamma=arg.lr_factor
    )

loss_func = nn.MSELoss()
def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return loss_func(model_out.flatten(), dataset_y.flatten())

metric_func = nn.L1Loss()
def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return -metric_func(model_out.flatten(), dataset_y.flatten()) / CHEMICAL_ACC_NORMALISING_FACTORS[target]

def get_dataset():
    root = ".dataset/QM9"
    transorm = tgtrans.Compose([
        lambda x: data_process.process(x, directional_tree=False),
    ])
    train = QM9Dataset(root=root, split="train", pre_transform=transorm)
    valid = QM9Dataset(root=root, split="valid", pre_transform=transorm)
    test  = QM9Dataset(root=root, split="test", pre_transform=transorm)
    train.data.y = train.data.y[:, target]
    valid.data.y = valid.data.y[:, target]
    test.data.y = test.data.y[:, target]
    return (train, valid, test)

def main():
    exec_config = ExecutionConfig(
        num_epoch = arg.total_epoch,
        batch_size = arg.batch_size,
        seed = arg.seed,
        save_cp_freq=1
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