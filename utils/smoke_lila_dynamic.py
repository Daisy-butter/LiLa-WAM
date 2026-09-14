#!/usr/bin/env python3
"""Smoke-test DOMINO + flow_cache loading for LiLa-Dynamic.

Usage:
  python utils/smoke_lila_dynamic.py
  python utils/smoke_lila_dynamic.py --config ./configs/lila_dynamic.yaml --index 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("smoke_lila_dynamic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lila_dynamic.yaml")
    parser.add_argument("--index", type=int, default=64, help="sample index (prefer >=48 for full history)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from dataloader.dataset import create_dataset, collate_fn

    cfg = OmegaConf.load(args.config)
    ds = create_dataset(cfg, val=False)
    logger.info(
        "dataset: n=%d episodes=%d layout=%s flow_cache=%s",
        len(ds),
        len(ds.episode_metadata),
        ds.layout,
        ds.flow_cache_root,
    )
    assert ds.layout == "dynamicwam_raw", ds.layout
    assert ds.flow_cache_root is not None
    assert len(ds.motion_stats["mean"]) == 12

    idx = min(max(args.index, 0), len(ds) - 1)
    sample = ds[idx]
    assert sample is not None

    expected = {
        "pixel_values": (3, 240, 320),
        "flow_pixel_values": (4, 3, 64, 64),
        "motion_features": (4, 12),
        "motion_interval_valid": (4,),
        "motion_acceleration_valid": (4,),
        "action_sequence": (16, 14),
        "state": (1, 14),
        "future_pixel_values": (3, 240, 320),
        "task_cond": (1024,),  # DINOv3-L hidden; may differ if encoder changes
    }
    for key, shape in expected.items():
        assert key in sample, f"missing {key}"
        got = tuple(sample[key].shape)
        if key == "task_cond":
            assert sample[key].ndim == 1 and sample[key].numel() > 0, got
        else:
            assert got == shape, f"{key}: expected {shape}, got {got}"
        logger.info("%s %s dtype=%s", key, got, sample[key].dtype)

    batch = collate_fn([sample, sample])
    assert batch["flow_pixel_values"].shape == (2, 4, 3, 64, 64)
    assert batch["motion_features"].shape == (2, 4, 12)
    logger.info("collate OK: flow_pixel_values=%s", tuple(batch["flow_pixel_values"].shape))
    logger.info("SMOKE PASS")


if __name__ == "__main__":
    main()
