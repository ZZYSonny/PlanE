from utils.common_imports_experiment import *

arg = wandb_init()
if arg is None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--dim_plane_pe', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--flags_layer', type=str, default="plane")
    parser.add_argument('--flags_plane_agg', type=str, default="n_t_b_gr_cr")
    parser.add_argument('--flags_mlp_factor', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--drop_loc', type=str, default="com")
    parser.add_argument('--lr_start', type=float, default=1e-3)
    parser.add_argument('--lr_end', type=float, default=1e-5)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=25)
    parser.add_argument('--total_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=str, default="ogbg-molhiv")
    arg = parser.parse_args()

transorm = tgtrans.Compose([
    data_process.process,
])
dataset = PygGraphPropPredDataset(arg.target, root = f".dataset", pre_transform=transorm)

def get_model():
    drop_loc = arg.drop_loc.split("_")
    model_config = models.ModelConfig(
        dim = arg.dim_hidden,
        dim_plane_pe = arg.dim_plane_pe,
        num_layers = arg.num_layers,
        dim_node_feature = "ogb_atom_node",
        dim_edge_feature = "ogb_atom_edge",
        dim_output = dataset.num_tasks,
        flags_layer=arg.flags_layer,
        flags_plane_agg=arg.flags_plane_agg,
        drop_enc=arg.drop if "enc" in drop_loc else 0,
        drop_rec=arg.drop if "rec" in drop_loc else 0,
        drop_agg=arg.drop if "agg" in drop_loc else 0,
        drop_com=arg.drop if "com" in drop_loc else 0,
        drop_out=arg.drop if "out" in drop_loc else 0,
        drop_edg=arg.drop if "edg" in drop_loc else 0,
        #act_out="Sigmoid"
    )
    return models.ModelGraph(model_config)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=arg.lr_start)

def get_scheduler(optimizer):
    return None

loss_func = torch.nn.BCEWithLogitsLoss()
def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    mask_not_nan = torch.logical_not(torch.isnan(dataset_y))
    return loss_func(model_out[mask_not_nan], dataset_y.float()[mask_not_nan])

evaluator = Evaluator(name=arg.target)
def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    res_dict = evaluator.eval({
        'y_true': dataset_y.numpy(),
        'y_pred': model_out.numpy(),
    })
    if 'ap' in res_dict:
        return res_dict["ap"]
    else:
        return res_dict["rocauc"]

def get_dataset():
    index = dataset.get_idx_split()
    return (
        dataset[index["train"]],
        dataset[index["valid"]],
        dataset[index["test"]],
    )

def main():
    exec_config = ExecutionConfig(
        num_epoch = arg.total_epoch,
        batch_size = arg.batch_size,
        seed = arg.seed,
        save_cp_freq=10,
        eval_train_freq=10,
        #log_loss_freq=1
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