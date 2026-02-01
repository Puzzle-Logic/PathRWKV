import os
import math
import time
import timm
import torch
import threading
import polars as pl
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Union
import torch.multiprocessing as mp
import torchvision.transforms.v2 as T
from safetensors.torch import save_file
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

save_executor = ThreadPoolExecutor(max_workers=1)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
torch.set_float32_matmul_precision("high")


class StreamingTileDataset(IterableDataset):
    def __init__(self, slide_ids, input_dir, transform=None):
        self.slide_ids = slide_ids
        self.input_dir = Path(input_dir)
        self.transform = transform
        self._df_cache = None

    def _get_df(self):
        if self._df_cache is None:
            self._df_cache = pl.read_csv(self.input_dir / "dataset.csv")
        return self._df_cache

    def process_slide(self, slide_id):
        try:
            df = self._get_df().filter(pl.col("slide_id") == slide_id)
            paths = df["tile_image_path"].to_list()
            ys = df["tile_y"].to_list()
            xs = df["tile_x"].to_list()

            total_tiles = len(paths)

            for i in range(total_tiles):
                img_path = self.input_dir / paths[i]
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))

                if self.transform:
                    img = self.transform(img)

                yield img, ys[i], xs[i], slide_id, total_tiles

        except Exception as e:
            print(f"Error reading slide {slide_id}: {e}")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_slides = self.slide_ids
        else:
            per_worker = int(
                math.ceil(len(self.slide_ids) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.slide_ids))
            my_slides = self.slide_ids[iter_start:iter_end]

        for slide_id in my_slides:
            yield from self.process_slide(slide_id)


def get_model(model_name, pretrained, device=None, compile_model=True):
    model = timm.create_model(
        model_name,
        num_classes=0,
        pretrained=pretrained if pretrained is True else False,
        checkpoint_path=pretrained if isinstance(pretrained, str) else None,
    )
    model = model.eval().to(device)
    model = torch.compile(model) if compile_model else model
    return model


def monitor_global_progress(global_bar, global_counter, total_slides, stop_event):
    while not stop_event.is_set():
        current = global_counter.value
        if global_bar.n < current:
            global_bar.n = current
            global_bar.refresh()
        if current >= total_slides:
            break
        time.sleep(0.5)
    global_bar.n = global_counter.value
    global_bar.refresh()


def flush_buffer_async(
    buffer_features, buffer_coords, slide_name, output_dir, global_counter
):
    save_path = output_dir / f"{slide_name}.safetensors"
    final_features = torch.cat(buffer_features, dim=0)
    final_coords = torch.cat(buffer_coords, dim=0)

    def _save_task(feat, coord, path, counter):
        save_file({"features": feat, "coords_yx": coord}, path)
        with counter.get_lock():
            counter.value += 1

    save_executor.submit(
        _save_task, final_features, final_coords, save_path, global_counter
    )


def run_worker(rank, slide_chunks, args, global_counter, total_slides, num_gpus):
    device_id = rank
    device = torch.device(f"cuda:{device_id}")
    my_slide_ids = slide_chunks[rank]
    input_dir = Path(args.input_dir)

    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = get_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
        compile_model=not args.disable_compile,
    )
    global_bar = None
    monitor_thread = None
    stop_monitor = threading.Event()

    if rank == 0:
        global_bar = tqdm(
            total=total_slides,
            desc="Total Progress",
            position=0,
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        monitor_thread = threading.Thread(
            target=monitor_global_progress,
            args=(global_bar, global_counter, total_slides, stop_monitor),
        )
        monitor_thread.start()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending_slides_for_this_gpu = []
    for s in my_slide_ids:
        if (output_dir / f"{s}.safetensors").exists():
            with global_counter.get_lock():
                global_counter.value += 1
        else:
            pending_slides_for_this_gpu.append(s)

    if not pending_slides_for_this_gpu:
        if rank == 0:
            while global_counter.value < total_slides:
                time.sleep(1)

            stop_monitor.set()
            monitor_thread.join()
            global_bar.close()

        return

    dataset = StreamingTileDataset(
        pending_slides_for_this_gpu, input_dir, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        if args.num_workers > 0
        else os.cpu_count() // num_gpus,
        pin_memory=True,
        drop_last=False,
    )

    slide_buffers = {}

    with torch.inference_mode():
        for batch in dataloader:
            images, ys, xs, slide_ids, total_tiles_batch = batch
            images = images.to(device, non_blocking=True)
            inference_dtype = torch.bfloat16 if args.bf16 else torch.float16
            print(f"Using {inference_dtype} precision")
            with torch.autocast(device_type="cuda", dtype=inference_dtype):
                features = model(images)
            features_cpu = features.cpu()
            unique_slides = set(slide_ids)

            for s_name in unique_slides:
                if len(unique_slides) == 1:
                    mask = slice(None)
                    count_in_batch = len(images)
                    total_tiles = total_tiles_batch[0].item()
                else:
                    mask = [s == s_name for s in slide_ids]
                    mask_tensor = torch.tensor(mask)
                    count_in_batch = sum(mask)
                    idx = mask.index(True)
                    total_tiles = total_tiles_batch[idx].item()

                if s_name not in slide_buffers:
                    slide_buffers[s_name] = {
                        "features": [],
                        "coords": [],
                        "count": 0,
                        "total": total_tiles,
                    }

                if len(unique_slides) == 1:
                    slide_buffers[s_name]["features"].append(features_cpu)
                    slide_buffers[s_name]["coords"].append(torch.stack([ys, xs], dim=1))
                else:
                    slide_buffers[s_name]["features"].append(features_cpu[mask_tensor])
                    slide_buffers[s_name]["coords"].append(
                        torch.stack([ys[mask_tensor], xs[mask_tensor]], dim=1)
                    )

                slide_buffers[s_name]["count"] += count_in_batch

                if slide_buffers[s_name]["count"] >= slide_buffers[s_name]["total"]:
                    flush_buffer_async(
                        slide_buffers[s_name]["features"],
                        slide_buffers[s_name]["coords"],
                        s_name,
                        output_dir,
                        global_counter,
                    )
                    del slide_buffers[s_name]

    save_executor.shutdown(wait=True)

    if rank == 0:
        stop_monitor.set()
        monitor_thread.join()
        global_bar.close()
        print("\nProcessing complete.")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("Scanning slides...")
    main_csv = input_dir / "dataset.csv"
    df = pl.read_csv(main_csv)
    all_slide_ids = sorted(df["slide_id"].unique().to_list())
    total_slides = len(all_slide_ids)

    processed_ids = set()
    if output_dir.exists():
        processed_ids = {f.stem for f in output_dir.glob("*.safetensors")}

    pending_slide_ids = [s for s in all_slide_ids if s not in processed_ids]
    num_pending = len(pending_slide_ids)

    print(
        f"Total: {total_slides}, Processed: {len(processed_ids)}, Pending: {num_pending}"
    )

    if num_pending == 0:
        print("All slides processed.")
        return

    if args.devices == -1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = args.devices

    chunk_size = math.ceil(num_pending / num_gpus)
    slide_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_pending)
        if start < num_pending:
            slide_chunks.append(pending_slide_ids[start:end])
        else:
            slide_chunks.append([])

    global_counter = mp.Value("i", total_slides - num_pending)

    mp.spawn(
        run_worker,
        args=(slide_chunks, args, global_counter, total_slides, num_gpus),
        nprocs=num_gpus,
        join=True,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument(
        "--model_name", type=str, default="hf_hub:prov-gigapath/prov-gigapath"
    )
    parser.add_argument("--pretrained", type=Union[str, bool], default=True)
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--bf16", action="store_false")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    main(args)
