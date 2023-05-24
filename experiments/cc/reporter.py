from utils.common_imports_experiment import *
from utils.processing.datasets import QM9Dataset
import os

run_ids = [
]

for run_id in run_ids:
    run = wandb.init(project='PlanE', id=run_id, resume="must")
    arg = wandb.run.config

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
    root = ".dataset/QM9CC"
    model = models.ModelGraph(model_config)
    test  = QM9Dataset(root=root, split="test")
    test.data.y = (test.data.y - 0.06) * 10
    loader = tgloader.DataLoader(test, batch_size=256)

    state_path = f".model_state/{run_id}_best"
    checkpoint = torch.load(state_path)
    model.load_state_dict(checkpoint)
    model.eval()

    out = []
    for data in loader:
        pred = model(data)
        out.append(pred.detach())

    out = torch.cat(out, dim=0).flatten()

    import pyperclip as pc
    a = [str(i) for i in out.tolist()]
    pc.copy("\n".join(a))
    print("Copied")
    wandb.finish()
    input("Press Enter to continue...")