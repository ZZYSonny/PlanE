import argparse
from datasets.helper import get_dataset, get_dataset_info
from experiments.trainer import *


def fn_get_model():
    from plane import models

    model_config = models.ModelConfig(
        dim=arg.model_dim_hidden,
        dim_plane_pe=arg.plane_dim_pe,
        num_layers=arg.model_num_layers,
        act_out=arg.model_final_act,
        flags_layer=arg.model_flags_layer,
        drop_agg=arg.model_dropout if "agg" in arg.model_dropout_loc else 0,
        drop_com=arg.model_dropout if "com" in arg.model_dropout_loc else 0,
        drop_enc=arg.model_dropout if "enc" in arg.model_dropout_loc else 0,
        drop_out=arg.model_dropout if "out" in arg.model_dropout_loc else 0,
        drop_rec=arg.model_dropout if "rec" in arg.model_dropout_loc else 0,
        drop_edg=arg.model_dropout if "edg" in arg.model_dropout_loc else 0,
        flags_plane_agg=arg.plane_flags_agg,
        flags_norm_before_com=arg.model_mlp_norm,
        flags_mlp_factor=arg.model_mlp_factor,
        flags_mlp_layer=arg.model_mlp_layer,
        **get_dataset_info(arg.dataset),
    )
    return models.ModelGraph(model_config)


def fn_get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=arg.lr_start)


def fn_get_scheduler(optimizer):
    if arg.lr_scheduler == "None":
        return None
    elif arg.lr_scheduler.startswith("ReduceLROnPlateau"):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=arg.metric_goal,
            min_lr=float(arg.lr_scheduler.split("_")[1]),
            factor=float(arg.lr_scheduler.split("_")[2]),
            patience=int(arg.lr_scheduler.split("_")[3]),
            verbose=True,
        )
    elif arg.lr_scheduler.startswith("StepLR"):
        return optim.lr_scheduler.StepLR(
            optimizer,
            gamma=float(arg.lr_scheduler.split("_")[1]),
            step_size=int(arg.lr_scheduler.split("_")[2]),
        )


def fn_get_dataset():
    # TO BE REMOVED
    from preprocess.data_process import process

    return get_dataset(
        name=arg.dataset, split=arg.cur_split, fn_final_transform=process
    )


if __name__ == "__main__":
    wandb_init()
    print("Using Arg Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cur_seed", type=int)
    parser.add_argument("--cur_split", type=str)
    parser.add_argument("--fn_loss", type=str)
    parser.add_argument("--fn_metric", type=str)
    parser.add_argument("--metric_goal", type=str)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--lr_start", type=float)
    parser.add_argument("--lr_scheduler", type=str)

    parser.add_argument("--model_flags_layer", type=str)
    parser.add_argument("--model_num_layers", type=int)
    parser.add_argument("--model_dim_hidden", type=int)
    parser.add_argument("--model_final_act", type=str)

    parser.add_argument("--model_dropout", type=float)
    parser.add_argument("--model_dropout_loc", type=str)

    parser.add_argument("--plane_dim_pe", type=int)
    parser.add_argument("--plane_flags_agg", type=str)

    parser.add_argument("--model_mlp_factor", type=int, default=2)
    parser.add_argument("--model_mlp_layer", type=str, default=2)
    parser.add_argument("--model_mlp_norm", type=str, default="batch_norm")

    parser.add_argument("--trainer_log_loss_freq", type=int, default=1)
    parser.add_argument("--trainer_save_cp_freq", type=int, default=1)
    parser.add_argument("--trainer_eval_train_freq", type=int, default=-1)
    parser.add_argument("--trainer_save_best_state", type=str, default="None")
    arg = parser.parse_args()

    match arg.fn_loss:
        case "L1":
            loss_func = nn.L1Loss()
            fn_loss = lambda pred, y: loss_func(pred.flatten(), y.flatten())
        case "L2":
            loss_func = nn.MSELoss()
            fn_loss = lambda pred, y: torch.sqrt(
                loss_func(pred.flatten(), y.flatten())
            )
        case "QM9_L2":
            loss_func = nn.MSELoss()
            fn_loss = lambda pred, y: loss_func(pred.flatten(), y.flatten())
        case "BCEWithLogitsLoss":
            loss_func = nn.BCEWithLogitsLoss()
            fn_loss = lambda pred, y: loss_func(
                pred.flatten(), y.float().flatten()
            )
        case "CrossEntropyLoss":
            fn_loss = nn.CrossEntropyLoss()
        case _:
            raise Exception("Unknown Loss Function")

    match arg.fn_metric:
        case "L1":
            loss_func = nn.L1Loss()
            fn_metric = lambda pred, y: loss_func(pred.flatten(), y.flatten())
        case "QM9_L1":
            from datasets.qm9 import CHEMICAL_ACC_NORMALISING_FACTORS

            target = int(arg.cur_split)
            loss_func = nn.L1Loss()
            factor = CHEMICAL_ACC_NORMALISING_FACTORS[target]
            fn_metric = (
                lambda pred, y: loss_func(pred.flatten(), y.flatten()) / factor
            )
        case "binary_thres_0_accuracy":
            fn_metric = (
                lambda pred, y: ((pred.flatten() > 0) == y.flatten())
                .float()
                .mean()
                .item()
            )
        case "multi_class_accuracy":
            fn_metric = (
                lambda pred, y: ((pred.argmax(dim=-1)) == y.flatten())
                .float()
                .mean()
                .item()
            )
        case "ogb":
            from datasets.ogb import ogb_evaluator_wrapper

            fn_metric = ogb_evaluator_wrapper(arg.dataset)
        case _:
            raise Exception("Unknown Metric Function")

    exec_config = ExecutionConfig(
        num_epoch=arg.epochs,
        batch_size=arg.batch_size,
        seed=arg.cur_seed,
        goal=arg.metric_goal,
        log_loss_freq=arg.trainer_log_loss_freq,
        save_cp_freq=arg.trainer_save_cp_freq,
        save_best_state=arg.trainer_save_best_state,
        eval_train_freq=arg.trainer_eval_train_freq,
    )

    Trainer(
        get_model=fn_get_model,
        get_optimizer=fn_get_optimizer,
        get_scheduler=fn_get_scheduler,
        get_dataset=fn_get_dataset,
        find_loss=fn_loss,
        find_metric=fn_metric,
        exec_config=exec_config,
    ).run()
