import math
import json
import torch
from torch import nn
from torch import optim
from pathlib import Path
from collections import defaultdict
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import (
    AUROC,
    Recall,
    F1Score,
    R2Score,
    Accuracy,
    Precision,
    CohenKappa,
    Specificity,
    PearsonCorrCoef,
    MetricCollection,
    MeanSquaredError,
    MatthewsCorrCoef,
    SpearmanCorrCoef,
    MeanAbsoluteError,
    ConcordanceCorrCoef,
)

from .utils import SksurvCIndex
from ..model.pathrwkv import PathRWKVv6 as PathRWKV


class Head(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.task_heads = nn.ModuleDict()

        for task in args.tasks.keys():
            if args.tasks[task]["type"] == "cls":
                num_classes = len(args.tasks[task]["labels"])
            else:
                num_classes = 1
            self.task_heads[task] = nn.Linear(input_dim, num_classes)

    def forward(self, backbone_output):
        outputs = {}
        for idx, task in enumerate(self.args.tasks.keys()):
            task_feat = backbone_output[:, idx, :]  # [B, D]
            task_out = self.task_heads[task](task_feat)
            outputs[task] = task_out

        return outputs  # {task_name: tensor, ...}


class WSIPipeline(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_model()

        if self.args.mode == "train" and not self.args.resume_ckpt:
            self.init_weights()

        args.data_path = str(args.data_path)
        args.runs_path = str(args.runs_path)
        self.save_hyperparameters(args)
        self.init_metrics()
        self.loss_functions = nn.ModuleDict(
            {k: v["loss_func"] for k, v in args.tasks.items()}
        )

    def init_model(self):
        self.backbone = PathRWKV(self.args)
        self.input_projector = nn.Sequential(
            nn.Linear(self.args.input_dim, self.backbone.input_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.head = Head(
            self.args,
            self.backbone.output_dim
            if hasattr(self.backbone, "output_dim")
            else self.args.input_dim,
        )

    def init_weights(self):
        def _init_layer_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_layer_weights)
        if hasattr(self, "MTL_tokens"):
            nn.init.normal_(self.MTL_tokens, std=0.02)

    def configure_optimizers(self):
        def _lr_lambda(current_epoch, epochs, lrf):
            return ((1 + math.cos(current_epoch * math.pi / epochs)) / 2) * (
                1 - lrf
            ) + lrf

        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: _lr_lambda(epoch, self.args.epochs, self.args.lrf),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "Val/Loss",
            },
        }

    def compute_loss(self, outputs, targets):
        total_loss = 0.0
        for task, task_info in self.args.tasks.items():
            pred = outputs[task]
            target = targets[task]
            task_type = task_info["type"]

            if task_type == "cls":
                valid_idxs = torch.where(target != -1)[0]

            elif task_type == "reg":
                pred, target = pred.view(-1), target.view(-1)
                valid_idxs = torch.where(target != -1.0)[0]

            elif task_type == "surv":
                valid_idxs = torch.where(target[:, 0] != -1.0)[0]

            if len(valid_idxs) > 0:
                task_loss = self.loss_functions[task](
                    pred[valid_idxs], target[valid_idxs]
                )
                total_loss += task_loss

        return total_loss

    def init_metrics(self):
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()

        for task_name, task_info in self.args.tasks.items():
            task_type = task_info["type"]

            if task_type == "cls":
                num_classes = len(task_info["labels"])
                metrics = MetricCollection(
                    {
                        "Kappa": CohenKappa(task="multiclass", num_classes=num_classes),
                        "MCC": MatthewsCorrCoef(
                            task="multiclass", num_classes=num_classes
                        ),
                        "AUC": AUROC(
                            task="multiclass", num_classes=num_classes, average="macro"
                        ),
                        "F1": F1Score(
                            task="multiclass", num_classes=num_classes, average="macro"
                        ),
                        "Recall": Recall(
                            task="multiclass", num_classes=num_classes, average="macro"
                        ),  # AKA Sensitivity
                        "Accuracy": Accuracy(
                            task="multiclass", num_classes=num_classes, average="micro"
                        ),
                        "Precision": Precision(
                            task="multiclass", num_classes=num_classes, average="macro"
                        ),
                        "Specificity": Specificity(
                            task="multiclass", num_classes=num_classes, average="macro"
                        ),
                        "per_class_AUC": AUROC(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                        "per_class_F1": F1Score(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                        "per_class_Recall": Recall(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                        "per_class_Accuracy": Accuracy(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                        "per_class_Precision": Precision(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                        "per_class_Specificity": Specificity(
                            task="multiclass", num_classes=num_classes, average="none"
                        ),
                    }
                )
            elif task_type == "reg":
                metrics = MetricCollection(
                    {
                        "R2": R2Score(),
                        "MSE": MeanSquaredError(),
                        "MAE": MeanAbsoluteError(),
                        "Pearson": PearsonCorrCoef(),  # AKA Correlation Coefficient
                        "CCC": ConcordanceCorrCoef(),
                        "Spearman": SpearmanCorrCoef(),
                    }
                )
            elif task_type == "surv":
                metrics = MetricCollection(
                    {
                        "C_Index": SksurvCIndex(),
                    }
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.val_metrics[task_name] = metrics.clone()
            self.test_metrics[task_name] = metrics.clone()

    def update_metrics(self, metrics_dict, preds, targets):
        for task_name, task_metrics in metrics_dict.items():
            task_pred = preds[task_name]
            task_target = targets[task_name]
            task_type = self.args.tasks[task_name]["type"]

            valid_idxs = None

            if task_type == "cls":
                valid_idxs = torch.where(task_target != -1)[0]
            elif task_type == "reg":
                valid_idxs = torch.where(task_target.reshape(-1) != -1.0)[0]
            elif task_type == "surv":
                valid_idxs = torch.where(task_target[:, 0] != -1.0)[0]

            if valid_idxs is None or len(valid_idxs) == 0:
                continue

            task_pred = task_pred[valid_idxs]
            task_target = task_target[valid_idxs]

            if task_type == "cls":
                task_pred_probs = torch.softmax(task_pred, dim=1)
                task_target_indices = task_target

                if "AUC" in task_metrics:
                    task_metrics["AUC"].update(task_pred_probs, task_target_indices)
                if "per_class_AUC" in task_metrics:
                    task_metrics["per_class_AUC"].update(
                        task_pred_probs, task_target_indices
                    )

                for key in task_metrics.keys():
                    if "AUC" not in key:
                        task_metrics[key].update(task_pred, task_target_indices)

            elif task_type == "reg":
                task_pred = task_pred.reshape(-1)
                task_target = task_target.reshape(-1)
                task_metrics.update(task_pred, task_target)

            elif task_type == "surv":
                times = task_target[:, 0]
                events = task_target[:, 1]
                task_metrics.update(task_pred, times, events)

    def _format_print(self, all_results, phase):
        print(f"\n=== {phase} Metrics ===")
        grouped_data = defaultdict(lambda: defaultdict(lambda: {}))
        task_names = sorted(list(self.args.tasks.keys()), key=len, reverse=True)

        for key, value in all_results.items():
            if not key.startswith(f"{phase}/"):
                continue

            content = key[len(phase) + 1 :]
            current_task = None
            for t_name in task_names:
                if content.startswith(t_name + "_"):
                    current_task = t_name
                    break

            if current_task is None:
                continue

            remainder = content[len(current_task) + 1 :]
            valid_metrics = list(self.val_metrics[current_task].keys())
            base_metrics = [m.replace("per_class_", "") for m in valid_metrics]
            possible_metrics = sorted(list(set(base_metrics)), key=len, reverse=True)

            metric_name = None
            class_name = "Val"
            for m in possible_metrics:
                if remainder.endswith("_" + m):
                    metric_name = m
                    break

            if metric_name:
                class_name = remainder[: -len(metric_name) - 1]
            else:
                parts = remainder.split("_")
                metric_name = parts[-1]
                if len(parts) > 1:
                    class_name = "_".join(parts[:-1])

            row_key = metric_name
            grouped_data[current_task][row_key][class_name] = value

        for task, metrics in grouped_data.items():
            print(f"\n[Task: {task}]")
            sorted_metrics = sorted(
                metrics.items(), key=lambda x: (len(x[1]) > 1, x[0])
            )
            lines_buffer = []

            for row_key, class_values in sorted_metrics:
                row_segments = [f"{row_key}:"]
                if "All" in class_values:
                    val = class_values["All"]
                    val_str = (
                        f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                    )
                    row_segments.append(f"All: {val_str}")

                other_classes = sorted([k for k in class_values.keys() if k != "All"])
                for i in other_classes:
                    val = class_values[i]
                    val_str = (
                        f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                    )

                    if i == "Val" and len(class_values) == 1:
                        row_segments.append(f"{val_str}")
                    else:
                        row_segments.append(f"{i}: {val_str}")

                lines_buffer.append(row_segments)

            if not lines_buffer:
                continue

            max_cols = max(len(row) for row in lines_buffer)
            col_widths = [0] * max_cols

            for row in lines_buffer:
                for idx, segment in enumerate(row):
                    col_widths[idx] = max(col_widths[idx], len(segment))

            for row in lines_buffer:
                formatted_row = []
                for idx, segment in enumerate(row):
                    if idx < len(row) - 1:
                        formatted_row.append(segment.ljust(col_widths[idx]))
                    else:
                        formatted_row.append(segment)

                print(" | ".join(formatted_row) + " |")

    def compute_metrics(self, metrics_dict, phase):
        all_results = {}
        for task_name, task_metrics in metrics_dict.items():
            task_results = task_metrics.compute()
            task_info = self.args.tasks[task_name]
            label_list = []

            if task_info["type"] == "cls":
                raw_labels = task_info.get("labels", [])
                if isinstance(raw_labels, dict):
                    name_to_index = {k: v for k, v in raw_labels.items()}
                    sorted_names = sorted(name_to_index, key=name_to_index.get)
                    label_list = sorted_names
                elif isinstance(raw_labels, list):
                    label_list = raw_labels
                else:
                    label_list = [f"Class_{i}" for i in range(len(raw_labels))]

            log_dict = {}
            for metric_name, value in task_results.items():
                if value.numel() == 1:
                    log_name = f"{phase}/{task_name}_All_{metric_name}"
                    val_item = value.item() if not torch.isnan(value) else 0.0

                    log_dict[log_name] = val_item
                    all_results[log_name] = round(val_item, 4)

                else:
                    base_metric_name = metric_name.replace("per_class_", "")
                    for i, v in enumerate(value):
                        class_name = (
                            label_list[i] if i < len(label_list) else f"Class_{i}"
                        )
                        log_name = (
                            f"{phase}/{task_name}_{class_name}_{base_metric_name}"
                        )

                        val_item = v.item() if not torch.isnan(v) else 0.0
                        log_dict[log_name] = val_item
                        all_results[log_name] = round(val_item, 4)

            self.log_dict(log_dict)
            task_metrics.reset()

        if self.global_rank == 0:
            self._format_print(all_results, phase)

        return all_results

    def forward(self, x, coords):  # Compatible with torch-based usage
        features = self.input_projector(x)
        preds = self.backbone(features, coords)
        preds = self.head(preds)
        return preds  # {task_name: tensor, ...}

    def shared_step(self, batch, phase):
        preds = self.forward(batch["image_features"], batch["coords_yx"])
        loss = self.compute_loss(preds, batch["task_labels"])
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        return loss if phase == "Train" else preds

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "Train")

    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, "Val")
        self.update_metrics(self.val_metrics, preds, batch["task_labels"])

    def on_validation_epoch_end(self):
        self.compute_metrics(self.val_metrics, "Val")

    def test_step(self, batch, batch_idx):
        preds = self.shared_step(batch, "Test")
        self.update_metrics(self.test_metrics, preds, batch["task_labels"])

    def on_test_epoch_end(self):
        results = self.compute_metrics(self.test_metrics, "Test")
        results_path = Path(self.args.runs_path) / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Test metrics saved to {results_path}")
