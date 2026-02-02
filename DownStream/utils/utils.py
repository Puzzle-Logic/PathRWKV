import yaml
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torchmetrics import Metric
from sksurv.metrics import concordance_index_censored


class CoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        times, events = targets[:, 0], targets[:, 1]
        sorted_idx = torch.argsort(times, descending=True)
        preds = preds[sorted_idx]
        events = events[sorted_idx]

        gamma = preds.max()
        exp_logits = torch.exp(preds - gamma)
        log_risk_set = torch.log(torch.cumsum(exp_logits, dim=0)) + gamma
        nll = -(preds - log_risk_set)
        masked_nll = nll * events
        num_events = events.sum()
        if num_events == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        return masked_nll.sum() / num_events


class SksurvCIndex(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")

    def update(self, preds, times, events):
        self.preds.append(preds.reshape(-1))
        self.times.append(times.reshape(-1))
        self.events.append(events.reshape(-1))

    def compute(self):
        preds = torch.cat(self.preds).detach().cpu().float().numpy()
        times = torch.cat(self.times).detach().cpu().float().numpy()
        events = torch.cat(self.events).detach().cpu().float().numpy()

        valid_mask = ~(np.isnan(preds) | np.isnan(times) | np.isnan(events))
        preds = preds[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]

        if len(preds) == 0:
            return torch.tensor(0.0)

        event_indicator = events.astype(bool)
        c_index = concordance_index_censored(event_indicator, times, preds)[0]
        return torch.tensor(c_index)


def initialize_experiment(args):
    # Get data path and input dimension
    input_dim = 1536  # Prov-GigaPath fixed dimension
    data_path = Path(args.data_path) / args.dataset_name / "tiles-embeddings"

    # Load task configuration
    task_config_path = (
        Path(__file__).resolve().parents[1]
        / "dataset_configs"
        / args.dataset_name
        / "task_configs.yaml"
    )
    with open(task_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_types = config["all_task_dict"]
    label_dict = config["label_dict"]

    if args.tasks is None:
        tasks_to_run = config["tasks_to_run"]
    elif isinstance(args.tasks, str):
        if "%" in args.tasks:
            tasks_to_run = args.tasks.split("%")
        else:
            tasks_to_run = [args.tasks]
    else:
        raise NotImplementedError(
            f"Unsupported type for tasks_to_run: {type(args.tasks)}"
        )

    def get_loss_func(task_type):
        if task_type == "cls":
            return nn.CrossEntropyLoss()
        elif task_type == "reg":
            return nn.SmoothL1Loss(beta=1.0)
        elif task_type == "surv":
            return CoxPHLoss()
        else:
            raise NotImplementedError(f"Unsupported task type: {task_type}")

    tasks = {}
    for task in tasks_to_run:
        tasks[task] = {
            "type": task_types[task],
            "labels": label_dict.get(task, None),
            "loss_func": get_loss_func(task_types[task]),
        }

    # Generate runs path and name
    tasks_str = "_".join(tasks.keys())
    runs_name = f"{tasks_str}_{args.lr:.0e}_B{args.batch_size}_T{args.max_tiles}"

    runs_path = Path(args.runs_path) / args.dataset_name / runs_name

    if args.mode == "train":
        runs_path.mkdir(parents=True, exist_ok=True)

    # Setup GPU devices
    if args.devices != "auto":
        devices = [int(num_str) for num_str in args.devices.split("%")]
    else:
        devices = args.devices

    return [data_path, input_dim, tasks, runs_path, runs_name, devices]
