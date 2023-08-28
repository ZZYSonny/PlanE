def ogbg_dataset_wrapper(name, root, split, pre_transform=None, transform=None):
    name = name.replace("_", "-")
    if split=="val": split = "valid"
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name, root, pre_transform=pre_transform, transform=transform)
    split_idx = dataset.get_idx_split()
    return dataset[split_idx[split]]

def ogb_evaluator_wrapper(name):
    name = name.replace("_", "-")
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name=name)

    def fn_metric_impl(pred, y):
        res_dict = evaluator.eval({
            'y_true': y.numpy(),
            'y_pred': pred.numpy(),
        })
        if 'ap' in res_dict:
            return res_dict["ap"]
        else:
            return res_dict["rocauc"]

    return fn_metric_impl