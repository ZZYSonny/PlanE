from utils.common_imports_experiment import *

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
    parser.add_argument('--lr_factor', type=float, default=0.7)
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    arg = parser.parse_args()


def get_model():
    model_config = models.ModelConfig(
        dim = arg.dim_hidden,
        dim_plane_pe = arg.dim_plane_pe,
        num_layers = arg.num_layers,
        dim_node_feature = [32],
        dim_edge_feature = 4,
        dim_output = 1,
        flags_layer=arg.flags_layer,
        flags_plane_agg=arg.flags_plane_agg,
        drop_agg=0,
        drop_com=0,
        drop_enc=0,
        drop_out=0,
        drop_rec=0,
        drop_edg=0
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

loss_func = nn.L1Loss()
def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return loss_func(model_out.flatten(), dataset_y.flatten())

def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return -loss_func(model_out.flatten(), dataset_y.flatten())

def get_dataset():
    root = ".dataset/ZINCFull"

    transorm = tgtrans.Compose([
        data_process.process,
    ])

    return (
        tgsets.ZINC(root=root, subset=False, split="train", pre_transform=transorm),
        tgsets.ZINC(root=root, subset=False, split="val", pre_transform=transorm),
        tgsets.ZINC(root=root, subset=False, split="test", pre_transform=transorm),
    )

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