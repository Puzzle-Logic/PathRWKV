import os
import math
import torch
import itertools
import polars as pl
from pathlib import Path
from functools import partial
from torch.utils.data import Dataset
from safetensors.torch import safe_open
from pytorch_lightning import LightningDataModule
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, DistributedSampler

from .CONSTS import DATASETS

MAX_WORKERS = os.cpu_count()


class SlideDataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.args = args
        self.mode = mode
        self._load_df()
        self._filter_data()

    def __len__(self):
        return self.df.height

    def _load_df(self):
        task_names = list(self.args.tasks.keys())
        split_target_key = "fold_information_5fold-1"
        self.slide_id_key = DATASETS[
            "TCGA" if "TCGA" in self.args.dataset_name else self.args.dataset_name
        ]["slide_id_key"]
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "dataset_configs"
            / self.args.dataset_name
            / "task_description.csv"
        )
        cols_needed = [self.slide_id_key, split_target_key, *task_names]
        df = pl.read_csv(csv_path, columns=cols_needed)

        task_cols = [pl.col(name).is_not_null() for name in task_names]
        df = df.filter(pl.col(split_target_key) == self.mode)
        if task_cols:
            df = df.filter(pl.any_horizontal(task_cols))

        self.df = df

    @staticmethod
    def _process_row(row_dict, data_root, slide_id_key):
        slide_id = row_dict[slide_id_key]
        slide_path = data_root / f"{slide_id}.safetensors"
        if slide_path.exists():
            new_row_data = row_dict.copy()
            new_row_data["data_path"] = slide_path
            return [new_row_data]
        return []

    def _filter_data(self):
        print(
            f"Initializing {self.args.dataset_name} {self.mode} split... Scanning files using {MAX_WORKERS} threads..."
        )
        worker_func = partial(
            SlideDataset._process_row,
            data_root=Path(self.args.data_path),
            slide_id_key=self.slide_id_key,
        )
        rows_to_process = self.df.to_dicts()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_list_of_lists = list(executor.map(worker_func, rows_to_process))

        processed_rows = list(itertools.chain.from_iterable(results_list_of_lists))
        self.df = pl.DataFrame(processed_rows)
        print(
            f"{self.mode} dataset initialized: {self.df.height} safetensors files found."
        )

    def _load_tiles(self, data_path):
        with safe_open(data_path, framework="pt", device="cpu") as f:
            image_features = f.get_tensor("features")
            coords_yx = f.get_tensor("coords_yx")
        return image_features, coords_yx

    def random_sampling(self, image_features, coords_yx):
        indices = torch.randperm(len(image_features))
        return image_features[indices], coords_yx[indices]

    def _sample_tiles(self, image_features, coords_yx):
        image_features, coords_yx = self.random_sampling(image_features, coords_yx)
        return image_features, coords_yx

    def _pad_or_truncate_tiles(self, image_features, coords_yx):
        N, D = image_features.shape
        if N >= self.args.max_tiles:
            image_features = image_features[: self.args.max_tiles]
            coords_yx = coords_yx[: self.args.max_tiles]
        else:
            pad_size = self.args.max_tiles - N
            image_features = torch.cat(
                (
                    image_features,
                    torch.zeros(
                        (pad_size, D),
                        dtype=image_features.dtype,
                        device=image_features.device,
                    ),
                ),
                dim=0,
            )
            coords_yx = torch.cat(
                (
                    coords_yx,
                    torch.zeros(
                        (pad_size, 2), dtype=coords_yx.dtype, device=coords_yx.device
                    ),
                ),
                dim=0,
            )

        return image_features, coords_yx

    def __getitem__(self, idx):
        item_row = self.df.row(idx, named=True)
        slide_path = item_row["data_path"]
        image_features, coords_yx = self._load_tiles(slide_path)

        if self.mode == "Train":
            image_features, coords_yx = self._sample_tiles(image_features, coords_yx)
            image_features, coords_yx = self._pad_or_truncate_tiles(
                image_features, coords_yx
            )

        task_labels = {}
        for task, task_info in self.args.tasks.items():
            label = item_row.get(task, None)
            if label is None or (isinstance(label, float) and math.isnan(label)):
                label = None
            else:
                label = str(label)

            if task_info["type"] == "cls":
                class_id = task_info["labels"].get(label, None)
                if class_id is not None:
                    task_labels[task] = torch.tensor(class_id, dtype=torch.long)
                else:
                    task_labels[task] = torch.tensor(-1, dtype=torch.long)

            elif task_info["type"] == "surv":  # Survival: "time|event"
                if label:
                    time_str, event_str = label.split("|")
                    time = float(time_str)
                    event = float(event_str)
                    task_labels[task] = torch.tensor([time, event], dtype=torch.float32)
                else:
                    task_labels[task] = torch.tensor([-1.0, -1.0], dtype=torch.float32)

            else:  # Regression
                if label:
                    task_labels[task] = torch.tensor(float(label), dtype=torch.float32)
                else:
                    task_labels[task] = torch.tensor(-1.0, dtype=torch.float32)

        return {
            "image_features": image_features,
            "coords_yx": coords_yx,
            "task_labels": task_labels,
        }


class data_module(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = SlideDataset(self.args, "Train")
            self.val_set = SlideDataset(self.args, "Val")
        elif stage == "test":
            self.test_set = SlideDataset(self.args, "Test")

    def _get_sampler(self, dataset, shuffle):
        if self.trainer and self.trainer._accelerator_connector.is_distributed:
            return DistributedSampler(dataset, shuffle=shuffle, seed=self.args.seed)
        return None

    def _create_dataloader(self, dataset, shuffle=False):
        sampler = self._get_sampler(dataset, shuffle)
        return DataLoader(
            dataset=dataset,
            pin_memory=True,
            sampler=sampler,
            drop_last=shuffle,
            num_workers=self.args.num_workers
            if self.args.num_workers >= 0
            else os.cpu_count() // len(self.args.devices),
            shuffle=(sampler is None and shuffle),
            persistent_workers=self.args.num_workers > 0,
            batch_size=self.args.batch_size if shuffle else 1,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_set, shuffle=False)
