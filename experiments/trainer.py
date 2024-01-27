import os
from plane.common_imports import *
import wandb


def mytqdm(xs, *args, **kwargs):
    if os.uname().release[0] == "6":
        return tqdm(xs, *args, **kwargs)
    else:
        return xs


def wandb_init():
    if "RESUMEID" in os.environ:
        run_id = os.environ["RESUMEID"]
        print(run_id)
        state_path = f".model_state/{run_id}"
        assert os.path.exists(state_path)
        wandb.init(
            id=run_id,
            resume="must",
            settings=wandb.Settings(_disable_stats=True),
        )
    else:
        wandb.init(settings=wandb.Settings(_disable_stats=True))
        wandb.run.log_code(".")

    if "seed" in wandb.config:
        print("Config from WanDB")
        return wandb.config
    else:
        print("Config from Args")
        return None


@dataclass
class ExecutionConfig:
    num_epoch: int
    batch_size: int
    goal: typing.Literal["min", "max"]
    seed: typing.Optional[int]

    log_loss_freq: int = -1
    save_cp_freq: int = -1
    save_best_state: str = ""
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory: bool = torch.cuda.is_available()
    num_workers: int = 1
    drop_last: bool = False
    eval_train_freq: int = -1


@dataclass
class Trainer:
    # Core
    get_model: typing.Callable[[], nn.Module]
    get_optimizer: typing.Callable[[nn.Module], torch.optim.Optimizer]
    get_scheduler: typing.Callable[
        [torch.optim.Optimizer], torch.optim.lr_scheduler.ReduceLROnPlateau
    ]
    get_dataset: typing.Callable[
        [], typing.Tuple[typing.Iterable, typing.Iterable, typing.Iterable]
    ]
    find_loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    find_metric: typing.Callable[[torch.Tensor, torch.Tensor], typing.Any]
    exec_config: ExecutionConfig

    def compile_model(self, train_loader):
        data = next(iter(train_loader))
        data = data.to(self.exec_config.device)
        self.model(data)
        self.model = tg.compile(self.model, dynamic=True)
        self.model(data)
        print("Model Compiled")

    def train_one_epoch(self, loader):
        self.model.train()

        loss_sum = 0
        for cur, batch in enumerate(
            mytqdm(loader, leave=False, desc="Training")
        ):
            model_out = self.model(batch.to(self.exec_config.device))

            self.optimizer.zero_grad()
            loss = self.find_loss(model_out, batch.y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach().cpu().item()
            if (
                self.exec_config.log_loss_freq > 0
                and cur % self.exec_config.log_loss_freq == 0
            ):
                wandb.log({"loss": loss_sum / self.exec_config.log_loss_freq})
                loss_sum = 0

            if self.epoch == 0 and cur == 0:
                p = sum(
                    p.numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                )
                print(f"Trainable Parameters: {p}")

    def evaluate_model(self, loader, task):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in mytqdm(loader, leave=False, desc=f"Eval {task}"):
                pred = self.model(batch.to(self.exec_config.device))
                y_true.append(batch.y.detach().cpu())
                y_pred.append(pred.detach().cpu())

            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            metric = self.find_metric(y_pred, y_true)
            return metric

    def to_device(self, datalist):
        return [data.to(self.exec_config.device) for data in datalist]

    def run(self):
        torch.manual_seed(self.exec_config.seed)
        self.state_path = f".model_state/{wandb.run.id}"
        self.model = self.get_model().to(self.exec_config.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer)

        epoch_start = 0
        best_metric = {
            name: float(-1e9) if self.exec_config.goal == "max" else float(1e9)
            for name in ["train", "valid", "test"]
        }
        train_dataset, valid_dataset, test_dataset = self.get_dataset()
        train_loader = tgloader.DataLoader(
            train_dataset,
            batch_size=self.exec_config.batch_size,
            shuffle=True,
            pin_memory=self.exec_config.pin_memory,
            num_workers=self.exec_config.num_workers,
            drop_last=self.exec_config.drop_last,
        )
        valid_loader = tgloader.DataLoader(
            valid_dataset,
            batch_size=self.exec_config.batch_size,
            shuffle=False,
            pin_memory=self.exec_config.pin_memory,
            num_workers=self.exec_config.num_workers,
        )
        test_loader = tgloader.DataLoader(
            test_dataset,
            batch_size=self.exec_config.batch_size,
            shuffle=False,
            pin_memory=self.exec_config.pin_memory,
            num_workers=self.exec_config.num_workers,
        )

        if wandb.run.resumed:
            print("Resuming")
            assert os.path.exists(self.state_path)
            checkpoint = torch.load(self.state_path)
            if self.model is not None:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
            if self.scheduler is not None:
                self.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )
            self.best_metric = checkpoint["best_metric"]
            epoch_start = checkpoint["epoch"] + 1
            torch.manual_seed(checkpoint["seed"])

        wandb.alert(title="Training Started", text="Training Started")
        bar = mytqdm(
            range(epoch_start, self.exec_config.num_epoch),
            leave=False,
            desc="Iter",
        )
        for epoch in bar:
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            if epoch == epoch_start:
                p = sum(
                    p.numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                )
                print(f"Trainable Parameters: {p}")
                wandb.run.summary["param"] = p

            cur_metric = {
                "epoch": epoch,
                "train": (
                    self.evaluate_model(train_loader, "train")
                    if self.exec_config.eval_train_freq > 0
                    and epoch % self.exec_config.eval_train_freq == 0
                    else -100
                ),
                "valid": self.evaluate_model(valid_loader, "valid"),
                "test": self.evaluate_model(test_loader, "test"),
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            if self.exec_config.save_best_state == "None":
                is_best = False
            else:
                criterion = self.exec_config.save_best_state
                if self.exec_config.goal == "max":
                    is_best = cur_metric[criterion] > best_metric[criterion]
                else:
                    is_best = cur_metric[criterion] < best_metric[criterion]

            for name in ["train", "valid", "test"]:
                if self.exec_config.goal == "max":
                    best_metric[name] = max(
                        best_metric[name], cur_metric[name]
                    )
                else:
                    best_metric[name] = min(
                        best_metric[name], cur_metric[name]
                    )

            wandb.log(cur_metric)
            wandb.run.summary.update(
                {
                    f"best_{name}": best_metric[name]
                    for name in ["train", "valid", "test"]
                }
            )

            if isinstance(bar, tqdm):
                bar.set_description(str(cur_metric))

            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(cur_metric["valid"])
            elif isinstance(
                self.scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.scheduler.step()

            if (
                self.exec_config.save_cp_freq != -1
                and epoch % self.exec_config.save_cp_freq == 0
            ):
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            self.model.state_dict()
                            if self.model is not None
                            else None
                        ),
                        "optimizer_state_dict": (
                            self.optimizer.state_dict()
                            if self.optimizer is not None
                            else None
                        ),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                        "best_metric": best_metric,
                        "seed": torch.seed(),
                    },
                    self.state_path,
                )

            if is_best:
                torch.save(self.model.state_dict(), self.state_path + "_best")

        wandb.alert("Training Finished", f"Best Test: {best_metric['test']}")
