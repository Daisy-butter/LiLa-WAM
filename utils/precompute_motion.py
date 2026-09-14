"""Optional offline Farnebäck cache builder (fallback only).

Prefer the official DynamicWAM cache already on disk:
  /SSD_DISK/users/wuruihan/wam/datasets/flow_cache/domino_absolute_motion_v2

Only run this if you need a custom cache for a different tree.

Usage:
  python utils/precompute_motion.py \
      --config ./configs/lila_dynamic.yaml \
      --output_dir /SSD_DISK/users/wuruihan/wam/datasets/flow_cache/lila_custom
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dataloader.dataset import _decode, create_dataset
from models.motion import (
    MOTION_FEATURE_DIM,
    MOTION_FEATURE_NAMES,
    build_motion_features,
    compute_flow_observation,
    interval_stride,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _episode_times(root, length: int, fps: float) -> np.ndarray:
    if "interception" in root and "sim_time_seconds" in root["interception"]:
        times = np.asarray(root["interception"]["sim_time_seconds"][()], dtype=np.float64)
        if times.shape[0] == length:
            return times
    return np.arange(length, dtype=np.float64) / max(float(fps), 1e-6)


def precompute_one(hdf5_path: str, camera_name: str, motion_cfg: dict) -> dict:
    stride = interval_stride(
        int(motion_cfg["policy_stride"]),
        int(motion_cfg["global_downsample_rate"]),
    )
    compute_size = tuple(motion_cfg["compute_size"])
    fps = float(motion_cfg["container_fps"])

    with h5py.File(hdf5_path, "r") as root:
        length = int(root["joint_action"]["vector"].shape[0])
        times = _episode_times(root, length, fps)
        rgb_ds = root["observation"][camera_name]["rgb"]

        flow_rgb = np.zeros((length, compute_size[0], compute_size[1], 3), dtype=np.uint8)
        displacement = np.zeros((length, 4), dtype=np.float32)
        starts = np.zeros(length, dtype=np.float64)
        interval_valid = np.zeros(length, dtype=np.bool_)
        history: deque = deque(maxlen=stride)
        first_rgb = None

        for idx in range(length):
            frame = _decode(rgb_ds[idx])
            if first_rgb is None:
                first_rgb = frame
            previous_idx = max(0, idx - stride)
            previous = first_rgb if idx < stride else history[0]
            starts[idx] = times[previous_idx]
            temporal_ok = (idx - previous_idx) == stride and times[idx] > times[previous_idx]
            if temporal_ok:
                rgb, stats, _frac, quality_valid = compute_flow_observation(
                    previous,
                    frame,
                    compute_size=compute_size,
                    normalization_percentile=float(motion_cfg["normalization_percentile"]),
                    farneback=motion_cfg.get("farneback") or {},
                    quality=motion_cfg.get("quality") or {},
                )
                interval_valid[idx] = bool(quality_valid)
                if quality_valid:
                    flow_rgb[idx] = rgb
                    displacement[idx] = stats
            history.append(frame)

    previous = np.arange(length, dtype=np.int64) - stride
    features, interval_valid, acceleration_valid = build_motion_features(
        displacement, starts, times, interval_valid, previous_interval_indices=previous,
    )
    return {
        "flow_rgb": flow_rgb,
        "motion_features": features,
        "interval_valid": interval_valid,
        "acceleration_valid": acceleration_valid,
    }


def _stats_terms(features, interval_valid, acceleration_valid):
    valid = np.repeat(interval_valid[:, None], MOTION_FEATURE_DIM, axis=1)
    valid[:, 9:12] = acceleration_valid[:, None]
    values = np.asarray(features, dtype=np.float64)
    count = valid.sum(axis=0, dtype=np.int64)
    total = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
    total_square = np.where(valid, values * values, 0.0).sum(axis=0, dtype=np.float64)
    return count, total, total_square


def main():
    parser = argparse.ArgumentParser(description="Precompute history-flow caches + motion stats")
    parser.add_argument("--config", type=str, default="./configs/lila_dynamic.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    motion_cfg = OmegaConf.to_container(cfg.model.motion, resolve=True)
    if not motion_cfg.get("enabled", False):
        logger.warning("model.motion.enabled is false; still precomputing with the configured geometry.")

    dataset = create_dataset(cfg, val=False)
    camera_name = list(cfg.dataset.camera_names)[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = np.zeros(MOTION_FEATURE_DIM, dtype=np.int64)
    total = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)
    total_square = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)

    seen = set()
    for ep in tqdm(dataset.all_episodes, desc="Precomputing motion"):
        hdf5_path = ep["hdf5_path"]
        if hdf5_path in seen:
            continue
        seen.add(hdf5_path)
        arrays = precompute_one(hdf5_path, camera_name, motion_cfg)
        split = ep.get("split") or "all"
        task = ep["task_name"]
        episode_id = str(ep.get("episode_id", Path(hdf5_path).stem.removeprefix("episode")))
        out_path = output_dir / split / task / "videos" / f"{episode_id}.flow.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **arrays)
        c, t, ts = _stats_terms(
            arrays["motion_features"], arrays["interval_valid"], arrays["acceleration_valid"],
        )
        total_count += c
        total += t
        total_square += ts

    scale_floor = 1e-6
    mean = np.divide(total, np.maximum(total_count, 1), dtype=np.float64)
    variance = np.maximum(total_square / np.maximum(total_count, 1) - mean * mean, 0.0)
    std = np.sqrt(variance)
    scale = np.maximum(std, scale_floor)
    stats = {
        "schema_version": 2,
        "feature_names": list(MOTION_FEATURE_NAMES),
        "count": total_count.tolist(),
        "mean": mean.tolist(),
        "standard_deviation": std.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": scale_floor,
        "temporal_contract": "lila_wam_dynamicwam_motion_v1",
    }
    stats_path = output_dir / "motion_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(seen)} caches and {stats_path}")


if __name__ == "__main__":
    main()
