import sys
import torch
import argparse
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_PATH))

from DownStream.utils.pipeline import WSIPipeline
from DownStream.utils.dataset import data_module
from DownStream.utils.utils import initialize_experiment

torch.set_float32_matmul_precision("high")


def get_args_parser():
    parser = argparse.ArgumentParser(description="PathRWKV MIL Pipeline")

    group_env = parser.add_argument_group("Environment Settings")
    group_env.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    group_env.add_argument(
        "--devices",
        type=str,
        default="0",
        help="GPU IDs: '0', '0%1', or 'auto' for all available GPUs (default: '0')",
    )
    group_env.add_argument(
        "--runs_path",
        type=str,
        default=str(ROOT_PATH.parent / "runs"),
        help="Path to save logs/results",
    )
    group_env.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Running mode (default: 'train')",
    )

    group_data = parser.add_argument_group("Data Settings")
    group_data.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset root path",
    )
    group_data.add_argument(
        "--dataset_name",
        type=str,
        default="CAMELYON16",
        help="Dataset name (e.g., CAMELYON16, PANDA)",
    )
    group_data.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Specific tasks (split by %), default: run all in config",
    )
    group_data.add_argument(
        "--num_workers", type=int, default=-1, help="Dataloader workers"
    )
    group_data.add_argument(
        "--max_tiles",
        type=int,
        default=2000,
        help="Max tiles per slide duirng training (sampling/padding limit)",
    )

    group_model = parser.add_argument_group("Model Settings")
    group_model.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to load trained checkpoint for continue training, 'None' for init from scratch",
    )
    group_model.add_argument(
        "--test_ckpt",
        type=str,
        default=None,
        help="Path to load checkpoint for testing, 'None' for default best.ckpt in runs_path",
    )

    group_train = parser.add_argument_group("Training Settings")
    group_train.add_argument("--batch_size", type=int, default=4, help="Batch size")
    group_train.add_argument("--epochs", type=int, default=100, help="Total epochs")
    group_train.add_argument(
        "--val_interval",
        type=float,
        default=1.0,
        help="Validation interval, 1.0 for epoch, 1 for step",
    )
    group_train.add_argument(
        "--early_stop_epoch", type=int, default=10, help="Patience for early stopping"
    )
    group_train.add_argument(
        "--lr", type=float, default=1e-04, help="Learning rate (default: 1e-4)"
    )
    group_train.add_argument(
        "--lrf",
        type=float,
        default=0.1,
        help="Cosine annealing end learning rate ratio",
    )
    group_train.add_argument(
        "--disable_pbar", action="store_true", help="Disable tqdm progress bar"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    seed_everything(args.seed, workers=True)
    (
        args.data_path,
        args.input_dim,
        args.tasks,
        args.runs_path,
        args.runs_name,
        args.devices,
    ) = initialize_experiment(args)

    if args.mode == "train":
        tb_logger = TensorBoardLogger(
            version="tb",
            name=args.runs_name,
            default_hp_metric=False,
            save_dir=args.runs_path.parent,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.runs_path / "checkpoints",
            filename="best",
            monitor="Val/Loss",
            mode="min",
            save_top_k=1,
        )
        early_stop_callback = EarlyStopping(
            monitor="Val/Loss",
            min_delta=0.00001,
            patience=args.early_stop_epoch,
            verbose=True,
            mode="min",
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    else:
        tb_logger = False
        callbacks = None

    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        devices=args.devices,
        precision="bf16-mixed",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        enable_model_summary=True,
        val_check_interval=args.val_interval,
        enable_progress_bar=not args.disable_pbar,
        strategy="ddp" if len(args.devices) != 1 else "auto",
    )

    dm = data_module(args)

    if args.mode == "train":
        model = WSIPipeline(args)
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)
        model = WSIPipeline.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path, args=args, weights_only=False
        )
        trainer.test(model, datamodule=dm)

    else:  # test
        ckpt_path = (
            args.test_ckpt
            if args.test_ckpt
            else args.runs_path / "checkpoints" / "best.ckpt"
        )
        print(f"Loading checkpoint from {ckpt_path}")
        model = WSIPipeline.load_from_checkpoint(
            ckpt_path, args=args, weights_only=False
        )
        trainer.test(model, datamodule=dm)
