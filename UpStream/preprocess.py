import os
import gc
import logging
import functools
import traceback
import numpy as np
import polars as pl
from tqdm import tqdm
import multiprocessing
import skimage.filters
from pathlib import Path
import skimage.morphology
from scipy import ndimage
from PIL import Image, ImageDraw
from dataclasses import dataclass
from monai.data.wsi_reader import WSIReader
from typing import Tuple, Dict, Optional, Sequence, List, Union


LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SLIDE_SUFFIXES = [".svs", ".ndpi", ".tiff", ".tif", ".png"]


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"Width must be strictly positive, received {self.w}")
        if self.h <= 0:
            raise ValueError(f"Height must be strictly positive, received {self.h}")

    def __add__(self, shift: Sequence[int]) -> "Box":
        if len(shift) != 2:
            raise ValueError("Shift must be two-dimensional")
        return Box(x=self.x + shift[0], y=self.y + shift[1], w=self.w, h=self.h)

    def __mul__(self, factor: float) -> "Box":
        return Box(
            x=int(self.x * factor),
            y=int(self.y * factor),
            w=int(self.w * factor),
            h=int(self.h * factor),
        )

    def __rmul__(self, factor: float) -> "Box":
        return self * factor

    def __truediv__(self, factor: float) -> "Box":
        return self * (1.0 / factor)

    def merge(self, other: "Box") -> "Box":
        x1, y1 = min(self.x, other.x), min(self.y, other.y)
        x2 = max(self.x + self.w, other.x + other.w)
        y2 = max(self.y + self.h, other.y + other.h)
        return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    @staticmethod
    def from_slices(slices: Sequence[slice]) -> "Box":
        vert, horz = slices
        return Box(
            x=horz.start,
            y=vert.start,
            w=horz.stop - horz.start,
            h=vert.stop - vert.start,
        )


def _get_luminance(image_or_images: np.ndarray) -> np.ndarray:
    if image_or_images.ndim == 4:
        return np.dot(image_or_images.transpose(0, 2, 3, 1), LUMINANCE_WEIGHTS)
    elif image_or_images.ndim == 3:
        return np.dot(image_or_images.transpose(1, 2, 0), LUMINANCE_WEIGHTS)
    raise ValueError("Expected an image array with 3 or 4 dimensions.")


def _segment_foreground(image_or_images: np.ndarray) -> Tuple[np.ndarray, float]:
    luminance = _get_luminance(image_or_images)
    threshold = skimage.filters.threshold_otsu(luminance)
    foreground_mask = luminance < threshold
    foreground_mask = skimage.morphology.binary_closing(foreground_mask)
    return foreground_mask, threshold


def _build_tile_loc_y_x(y: int, x: int) -> str:
    return f"{y:05d}y_{x:05d}x"


def _get_tile_descriptor(tile_location: Sequence[int]) -> str:
    return _build_tile_loc_y_x(tile_location[0], tile_location[1])


def _get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    return f"{slide_id}.{_get_tile_descriptor(tile_location)}"


def _calculate_area(slice_obj: Tuple[slice, slice]) -> int:
    y_slice, x_slice = slice_obj
    return (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)


def _get_top_n_slices_by_size(mask: np.ndarray, maximum_top_n: int = 5) -> List:
    labeled_mask, _ = ndimage.label(mask > 0)
    slices = ndimage.find_objects(labeled_mask)
    if not slices:
        raise RuntimeError("No objects found in the mask")
    slice_areas = [(s, _calculate_area(s)) for s in slices]
    sorted_slices = sorted(slice_areas, key=lambda x: x[1], reverse=True)
    return [s for s, _ in sorted_slices[: min(maximum_top_n, len(sorted_slices))]]


def _merge_overlapping_boxes(box_list: List[Box]) -> List[Box]:
    merged_boxes = []
    box_list = list(box_list)
    while box_list:
        box = box_list.pop(0)
        merge_with = [
            b
            for b in box_list
            if (
                box.x < b.x + b.w
                and box.x + box.w > b.x
                and box.y < b.y + b.h
                and box.y + box.h > b.y
            )
        ]
        for b in merge_with:
            box = box.merge(b)
            box_list.remove(b)
        merged_boxes.append(box)
    return merged_boxes


def _align_box_to_grid(box: Box, grid_spacing: int, max_y: int, max_x: int) -> Box:
    x1 = (box.x // grid_spacing) * grid_spacing
    y1 = (box.y // grid_spacing) * grid_spacing
    x2 = ((box.x + box.w + grid_spacing - 1) // grid_spacing) * grid_spacing
    y2 = ((box.y + box.h + grid_spacing - 1) // grid_spacing) * grid_spacing
    return Box(
        x=max(0, x1),
        y=max(0, y1),
        w=min(max_x, x2) - max(0, x1),
        h=min(max_y, y2) - max(0, y1),
    )


def _check_an_empty_tile(
    tile: np.ndarray, t_occupancy: float, global_threshold: float = None
) -> Tuple[bool, float]:
    luminance = _get_luminance(tile)
    foreground_mask = luminance < global_threshold
    foreground_mask = skimage.morphology.binary_closing(foreground_mask)
    occupancy = float(foreground_mask.mean(axis=(-2, -1)))
    if occupancy < t_occupancy:
        return True, occupancy

    C, H, W = tile.shape
    flattened = tile.reshape(C, H * W)
    std_rgb_mean = float(flattened.std(axis=-1).mean())
    if std_rgb_mean < 5:
        return True, occupancy

    extreme_count = (flattened == 0).sum(axis=-1)
    extreme_max = float((extreme_count / (H * W)).max())
    return extreme_max > 0.5, occupancy


def _resize_tile_pil(tile_img: np.ndarray, tile_size: int) -> Image.Image:
    tile_hwc = np.transpose(tile_img, (1, 2, 0)).astype(np.uint8)
    pil_img = Image.fromarray(tile_hwc).convert("RGB")
    return pil_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)


def _save_pil_image(pil_image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.convert("RGB").save(str(path))


def _generate_rel_locations(tilesize: int, h: int, w: int) -> List[Tuple[int, int]]:
    num_y, num_x = h // tilesize, w // tilesize
    return [(i * tilesize, j * tilesize) for i in range(num_y) for j in range(num_x)]


def _adjust_tile_locations(
    rel_locs: List[Tuple[int, int]], scale: float, origin_y: int, origin_x: int
) -> np.ndarray:
    return np.array(
        [(int(y * scale) + origin_y, int(x * scale) + origin_x) for y, x in rel_locs],
        dtype=int,
    )


def _get_tile_info_dict(
    sample: Dict,
    occupancy: float,
    tile_location: Sequence[int],
    tilesize: int,
    rel_slide_dir: Path,
    ROI_image_key: str,
    suffix: str = ".jpeg",
) -> Dict:
    slide_id = sample["slide_id"]
    descriptor = _get_tile_descriptor(tile_location)
    return {
        "slide_id": slide_id,
        "tile_id": _get_tile_id(slide_id, tile_location),
        ROI_image_key: f"{rel_slide_dir}/{descriptor}{suffix}",
        "label": sample.get("label"),
        "tile_y": tile_location[0],
        "tile_x": tile_location[1],
        "target_level_tilesize": tilesize,
        "occupancy": occupancy,
        "metadata": {"slide_" + k: v for k, v in sample["metadata"].items()},
    }


def _format_csv_row(tile_info: Dict, keys: Tuple, metadata_keys: Tuple) -> str:
    metadata = tile_info.pop("metadata")
    fields = [str(tile_info[k]) for k in keys]
    fields.extend(str(metadata[k]) for k in metadata_keys)
    return ",".join(fields)


def _batch_write_csv(csv_file, rows: List[str], batch_size: int = 100) -> None:
    if rows:
        csv_file.write("\n".join(rows) + "\n")
        csv_file.flush()


def process_one_slide(
    sample: Dict,
    output_dir: Path = None,
    thumbnail_dir: Path = None,
    tile_size: int = 224,
    target_mpp: float = 0.5,
    manual_mpp: Optional[float] = None,
    force_read_level: Optional[int] = None,
    t_occupancy: float = 0.1,
    chunk_scale_in_tiles: int = 20,
    image_key: str = "slide_image_path",
    ROI_image_key: str = "tile_image_path",
    generate_thumbnails: bool = True,
) -> Optional[str]:
    reader = WSIReader(backend="OpenSlide")
    slide_id = sample["slide_id"]
    slide_path = Path(sample[image_key])
    slide_output_dir = output_dir / slide_id
    if slide_output_dir.exists():
        csv_path = slide_output_dir / "dataset.csv"
        if slide_output_dir.exists() and list(slide_output_dir.glob("*.jpeg")):
            try:
                df = pl.read_csv(csv_path)
                if len(df) > 0:
                    return None
            except Exception:
                pass

    slide_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        WSI_obj = reader.read(str(slide_path))
        finest_mpp = manual_mpp or float(reader.get_mpp(WSI_obj, level=0)[1])
        if finest_mpp is None:
            raise KeyError("MPP value missing")

        if force_read_level is not None:
            target_level = force_read_level
            ratio = WSI_obj.level_downsamples[target_level]
            nearest_mpp = finest_mpp * int(ratio)
        else:
            mpp_level_dict = {}
            for level in range(len(WSI_obj.level_downsamples)):
                ratio = WSI_obj.level_downsamples[level]
                mpp = finest_mpp * int(ratio)
                mpp_level_dict[mpp] = level

            if target_mpp in mpp_level_dict:
                target_level = mpp_level_dict[target_mpp]
                nearest_mpp = target_mpp
            elif target_mpp > finest_mpp:
                min_diff = float("inf")
                for mpp in mpp_level_dict:
                    diff = target_mpp - mpp if target_mpp > mpp else float("inf")
                    if diff < min_diff:
                        target_level, nearest_mpp, min_diff = (
                            mpp_level_dict[mpp],
                            mpp,
                            diff,
                        )
            else:
                min_diff = float("inf")
                for mpp in mpp_level_dict:
                    diff = mpp - target_mpp if mpp > target_mpp else float("inf")
                    if diff < min_diff:
                        target_level, nearest_mpp, min_diff = (
                            mpp_level_dict[mpp],
                            mpp,
                            diff,
                        )

        roi_scale = target_mpp / nearest_mpp

        highest_level = WSI_obj.level_count - 1
        slide_data, _ = reader.get_data(WSI_obj, level=highest_level)
        fg_mask, lum_threshold = _segment_foreground(slide_data)
        detection_scale = WSI_obj.level_downsamples[highest_level]
        level0_grid_spacing = tile_size * (target_mpp / finest_mpp)

        max_y = fg_mask.shape[0] * detection_scale
        max_x = fg_mask.shape[1] * detection_scale
        slices = _get_top_n_slices_by_size(fg_mask, maximum_top_n=20)
        box_list = [Box.from_slices(s) for s in slices]
        scaled_boxes = [detection_scale * box for box in box_list]
        level0_boxes = [
            _align_box_to_grid(box, int(level0_grid_spacing), max_y, max_x)
            for box in scaled_boxes
        ]
        level0_boxes = _merge_overlapping_boxes(level0_boxes)

        keys_to_save = (
            "slide_id",
            ROI_image_key,
            "tile_id",
            "label",
            "tile_y",
            "tile_x",
            "occupancy",
        )
        metadata_keys = tuple("slide_" + k for k in sample.get("metadata", {}))
        csv_columns = (*keys_to_save, *metadata_keys)

        csv_path = slide_output_dir / "dataset.csv"
        failed_path = slide_output_dir / "failed_tiles.csv"

        with open(csv_path, "w") as csv_file, open(failed_path, "w") as failed_file:
            csv_file.write(",".join(csv_columns) + "\n")
            failed_file.write("tile_id\n")

            for roi_idx, level0_box in enumerate(level0_boxes):
                if generate_thumbnails:
                    scale = 1024 / max(WSI_obj.dimensions)
                    thumb = WSI_obj.get_thumbnail(
                        [int(d * scale) for d in WSI_obj.dimensions]
                    )
                    draw = ImageDraw.Draw(thumb)
                    scaled_box = [
                        int(level0_box.x * scale),
                        int(level0_box.y * scale),
                        int((level0_box.x + level0_box.w) * scale),
                        int((level0_box.y + level0_box.h) * scale),
                    ]
                    draw.rectangle(scaled_box, outline="red", width=3)
                    thumb_path = (
                        thumbnail_dir / f"{slide_path.name}_original_ROI_{roi_idx}.jpeg"
                    )
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    thumb.save(str(thumb_path))

                wsi_scale = WSI_obj.level_downsamples[target_level]
                target_box = level0_box / wsi_scale
                tilesize = int(tile_size * roi_scale)

                target_h, target_w = target_box.h, target_box.w

                if chunk_scale_in_tiles <= 1:
                    chunk_locs = np.array([])
                    rel_locs = _generate_rel_locations(tilesize, target_h, target_w)
                    tile_locs = _adjust_tile_locations(
                        rel_locs, wsi_scale, level0_box.y, level0_box.x
                    )
                else:
                    chunk_size = tilesize * chunk_scale_in_tiles
                    all_locs = set(
                        _generate_rel_locations(tilesize, target_h, target_w)
                    )
                    chunk_h = target_h - target_h % chunk_size
                    chunk_w = target_w - target_w % chunk_size
                    chunk_region_locs = set(
                        _generate_rel_locations(tilesize, chunk_h, chunk_w)
                    )
                    remaining = list(all_locs - chunk_region_locs)

                    chunk_rel_locs = _generate_rel_locations(
                        chunk_size, chunk_h, chunk_w
                    )
                    chunk_locs = _adjust_tile_locations(
                        chunk_rel_locs, wsi_scale, level0_box.y, level0_box.x
                    )
                    tile_locs = _adjust_tile_locations(
                        remaining, wsi_scale, level0_box.y, level0_box.x
                    )

                tile_info_list = []
                csv_buffer = []
                n_discarded = 0
                CSV_BATCH_SIZE = 50

                for chunk_loc in chunk_locs:
                    level0_y, level0_x = chunk_loc
                    try:
                        chunk_img, _ = reader.get_data(
                            WSI_obj,
                            location=(level0_y, level0_x),
                            size=(
                                tilesize * chunk_scale_in_tiles,
                                tilesize * chunk_scale_in_tiles,
                            ),
                            level=target_level,
                        )
                        rel_tile_locs = _generate_rel_locations(
                            tilesize, chunk_img.shape[1], chunk_img.shape[2]
                        )
                        chunk_tile_locs = _adjust_tile_locations(
                            rel_tile_locs, wsi_scale, level0_y, level0_x
                        )

                        for (rel_y, rel_x), tile_loc in zip(
                            rel_tile_locs, chunk_tile_locs
                        ):
                            tile_img = chunk_img[
                                :, rel_y : rel_y + tilesize, rel_x : rel_x + tilesize
                            ]
                            is_empty, occ = _check_an_empty_tile(
                                tile_img, t_occupancy, lum_threshold
                            )
                            if is_empty:
                                n_discarded += 1
                            else:
                                pil_tile = _resize_tile_pil(tile_img, tile_size)
                                tile_info = _get_tile_info_dict(
                                    sample,
                                    occ,
                                    tile_loc,
                                    tilesize,
                                    Path(slide_id),
                                    ROI_image_key,
                                )
                                _save_pil_image(
                                    pil_tile,
                                    output_dir / tile_info[ROI_image_key],
                                )
                                tile_info_list.append(tile_info)
                                csv_buffer.append(
                                    _format_csv_row(
                                        tile_info.copy(), keys_to_save, metadata_keys
                                    )
                                )
                                if len(csv_buffer) >= CSV_BATCH_SIZE:
                                    _batch_write_csv(csv_file, csv_buffer)
                                    csv_buffer = []
                    except Exception:
                        failed_file.write(_get_tile_descriptor(chunk_loc) + "\n")
                        traceback.print_exc()

                for tile_loc in tile_locs:
                    level0_y, level0_x = tile_loc
                    try:
                        tile_img, _ = reader.get_data(
                            WSI_obj,
                            location=(level0_y, level0_x),
                            size=(tilesize, tilesize),
                            level=target_level,
                        )
                        is_empty, occ = _check_an_empty_tile(
                            tile_img, t_occupancy, lum_threshold
                        )
                        if is_empty:
                            n_discarded += 1
                        else:
                            pil_tile = _resize_tile_pil(tile_img, tile_size)
                            tile_info = _get_tile_info_dict(
                                sample,
                                occ,
                                tile_loc,
                                tilesize,
                                Path(slide_id),
                                ROI_image_key,
                            )
                            _save_pil_image(
                                pil_tile, output_dir / tile_info[ROI_image_key]
                            )
                            tile_info_list.append(tile_info)
                            csv_buffer.append(
                                _format_csv_row(
                                    tile_info.copy(), keys_to_save, metadata_keys
                                )
                            )
                            if len(csv_buffer) >= CSV_BATCH_SIZE:
                                _batch_write_csv(csv_file, csv_buffer)
                                csv_buffer = []
                    except Exception:
                        failed_file.write(_get_tile_descriptor(tile_loc) + "\n")
                        traceback.print_exc()

                if csv_buffer:
                    _batch_write_csv(csv_file, csv_buffer)
                    csv_buffer = []

                if tile_info_list and generate_thumbnails:
                    from matplotlib import collections, patches, pyplot as plt

                    vis_scale = 1024 / max(WSI_obj.dimensions)
                    vis_thumb = WSI_obj.get_thumbnail(
                        [int(d * vis_scale) for d in WSI_obj.dimensions]
                    )
                    downscale = 1 / vis_scale

                    fig, ax = plt.subplots()
                    ax.imshow(vis_thumb)
                    rects = []
                    for info in tile_info_list:
                        xy = (info["tile_x"] / downscale, info["tile_y"] / downscale)
                        ts = int(info["target_level_tilesize"] * wsi_scale / downscale)
                        rects.append(patches.Rectangle(xy, ts, ts))

                    pc = collections.PatchCollection(
                        rects, match_original=True, alpha=0.5, edgecolor="black"
                    )
                    pc.set_array(np.array([100] * len(tile_info_list)))
                    ax.add_collection(pc)
                    vis_path = (
                        thumbnail_dir / f"{slide_path.name}_roi_{roi_idx}_tiles.jpeg"
                    )
                    fig.savefig(str(vis_path))
                    plt.close()

        del WSI_obj
        gc.collect()
        return None

    except Exception as e:
        logging.error(f"Error processing slide {sample[image_key]}: {e}")
        return sample[image_key]


def gen_slides_list(
    input_dir: Union[str, Path],
    mode: Union[None, str] = None,
    metadata_dir: Optional[List[Union[str, Path]]] = None,
    image_key: str = "slide_image_path",
) -> List[Dict]:
    metadata_lookup = {}
    if metadata_dir and mode == "TCGA":
        print("Pre-loading metadata...")
        for meta_path in metadata_dir:
            try:
                if not os.path.exists(meta_path):
                    continue

                df = pl.read_csv(meta_path)
                if "patient_id" in df.columns:
                    for record in df.to_dicts():
                        if record["patient_id"]:
                            if record["patient_id"] not in metadata_lookup:
                                metadata_lookup[record["patient_id"]] = {}
                            metadata_lookup[record["patient_id"]].update(record)
            except Exception as e:
                print(f"Warning: Failed to load metadata {meta_path}: {e}")

    slides = []
    suffix_tuple = tuple(SLIDE_SUFFIXES)

    for root, _, files in os.walk(input_dir):
        for f in files:
            if "morphology" in f:
                continue

            if f.endswith(suffix_tuple):
                path = Path(root) / f
                slide_id = path.stem
                while "." in slide_id:
                    slide_id = slide_id.rsplit(".", 1)[0]

                patient_id = slide_id[:12] if mode == "TCGA" else slide_id
                sample = {image_key: str(path), "slide_id": slide_id, "metadata": {}}
                if mode == "TCGA" and patient_id in metadata_lookup:
                    sample["metadata"].update(metadata_lookup[patient_id])

                slides.append(sample)

    print(f"Found {len(slides)} slides.")
    return slides


def merge_csv(dataset_dir: Path) -> Path:
    full_csv = dataset_dir / "dataset.csv"
    all_dfs = []
    files_to_delete = []

    for slide_csv in dataset_dir.glob("*/dataset.csv"):
        df = pl.read_csv(slide_csv)
        all_dfs.append(df)
        files_to_delete.append(slide_csv)

    combined = pl.concat(all_dfs)
    combined.write_csv(full_csv)
    for f in files_to_delete:
        f.unlink()
    return full_csv


def process_all_slides(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mode: Union[None, str] = None,
    metadata_dir: Optional[List[Union[str, Path]]] = None,
    tile_size: int = 224,
    target_mpp: float = 0.5,
    manual_mpp: Optional[float] = None,
    force_read_level: Optional[int] = None,
    t_occupancy: float = 0.1,
    chunk_scale_in_tiles: int = 20,
    image_key: str = "slide_image_path",
    num_workers: Optional[int] = None,
    gen_thumbnails: bool = True,
) -> None:
    slides = gen_slides_list(
        input_dir,
        metadata_dir=metadata_dir,
        mode=mode,
        image_key=image_key,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if gen_thumbnails:
        thumbnail_dir = output_dir / "thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)

    func = functools.partial(
        process_one_slide,
        output_dir=output_dir,
        thumbnail_dir=thumbnail_dir if gen_thumbnails else None,
        tile_size=tile_size,
        target_mpp=target_mpp,
        manual_mpp=manual_mpp,
        force_read_level=force_read_level,
        t_occupancy=t_occupancy,
        chunk_scale_in_tiles=chunk_scale_in_tiles,
        image_key=image_key,
        generate_thumbnails=gen_thumbnails,
    )

    n_workers = num_workers or multiprocessing.cpu_count()
    chunksize = max(1, len(slides) // (n_workers * 4))
    with multiprocessing.Pool(
        processes=n_workers,
        maxtasksperchild=10,
    ) as pool:
        error_WSIs = list(
            tqdm(
                pool.imap_unordered(func, slides, chunksize=chunksize),
                desc="Slides",
                unit="WSI",
                total=len(slides),
            )
        )

    merge_csv(output_dir)

    errors = [e for e in error_WSIs if e is not None]
    if errors:
        logging.warning(f"Error WSIs: {errors}")
        print(f"Errors: {errors}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--metadata_dir", type=str, default=None)
    parser.add_argument("--edge_size", type=int, default=224)
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--manual_mpp", type=float, default=None)
    parser.add_argument("--force_read_level", type=int, default=None)
    parser.add_argument("--t_occupancy", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default=None, choices=["TCGA", None])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--gen_thumbnails", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_slides(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        metadata_dir=[args.metadata_dir] if args.metadata_dir else None,
        tile_size=args.edge_size,
        target_mpp=args.target_mpp,
        manual_mpp=args.manual_mpp,
        force_read_level=args.force_read_level,
        t_occupancy=args.t_occupancy,
        num_workers=args.num_workers,
        chunk_scale_in_tiles=args.chunk_size,
        gen_thumbnails=args.gen_thumbnails,
    )
