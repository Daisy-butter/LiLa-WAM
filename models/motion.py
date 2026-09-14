"""Dual-path motion primitives ported from DynamicWAM.

History-flow path: Farnebäck optical flow + forward-backward consistency,
rendered as RGB (direction -> hue, magnitude -> per-map 99th percentile).

Kinematic path: 12-D descriptors
  [mean_dx, mean_dy, mean_|d|, p99_|d|, dt,
   mean_vx, mean_vy, mean_speed, p99_speed,
   mean_ax, mean_ay, |a|]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


MOTION_FEATURE_NAMES = (
    "mean_displacement_x_pixels",
    "mean_displacement_y_pixels",
    "mean_displacement_magnitude_pixels",
    "p99_displacement_magnitude_pixels",
    "delta_t_seconds",
    "mean_velocity_x_pixels_per_second",
    "mean_velocity_y_pixels_per_second",
    "mean_speed_pixels_per_second",
    "p99_speed_pixels_per_second",
    "mean_acceleration_x_pixels_per_second2",
    "mean_acceleration_y_pixels_per_second2",
    "mean_acceleration_magnitude_pixels_per_second2",
)
MOTION_FEATURE_DIM = len(MOTION_FEATURE_NAMES)

FARNEBACK_DEFAULTS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

QUALITY_DEFAULTS = {
    "method": "forward_backward_consistency_v1",
    "relative_error": 0.01,
    "absolute_error_squared": 0.5,
    "minimum_reliable_fraction": 0.1,
}

IDENTITY_MOTION_STATS = {
    "mean": [0.0] * MOTION_FEATURE_DIM,
    "scale": [1.0] * MOTION_FEATURE_DIM,
}


def interval_stride(policy_stride: int, global_downsample_rate: int) -> int:
    policy_stride = int(policy_stride)
    global_downsample_rate = int(global_downsample_rate)
    if policy_stride <= 0 or global_downsample_rate <= 0:
        raise ValueError(
            "policy_stride and global_downsample_rate must be positive, "
            f"got {policy_stride}, {global_downsample_rate}"
        )
    return policy_stride * global_downsample_rate


def history_endpoint_offsets(history_count: int, stride: int) -> list[int]:
    """Offsets from the current frame to the K+1 interval endpoints (oldest first)."""
    history_count = int(history_count)
    stride = int(stride)
    if history_count <= 0 or stride <= 0:
        raise ValueError("history_count and stride must be positive")
    return [k * stride for k in range(history_count, -1, -1)]


def history_end_indices(anchor_idx: int, history_count: int, stride: int) -> list[int]:
    """Frame indices of the K interval *ends* (oldest to newest)."""
    return [max(0, int(anchor_idx) - k * int(stride)) for k in range(int(history_count) - 1, -1, -1)]


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for history-flow conditioning") from exc
    return cv2


def compute_dense_flow(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    compute_size: Tuple[int, int],
    farneback: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Signed Farnebäck displacement on the configured compute grid. Returns (H, W, 2)."""
    cv2 = _opencv()
    height, width = (int(v) for v in compute_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"compute_size must be positive, got {compute_size}")
    for name, frame in (("previous_rgb", previous_rgb), ("current_rgb", current_rgb)):
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[-1] != 3
        ):
            raise ValueError(
                f"{name} must be HWC uint8 RGB, got "
                f"{getattr(frame, 'dtype', None)} {getattr(frame, 'shape', None)}"
            )
    previous = cv2.resize(previous_rgb, (width, height), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_rgb, (width, height), interpolation=cv2.INTER_AREA)
    config = {**FARNEBACK_DEFAULTS, **dict(farneback or {})}
    flow = cv2.calcOpticalFlowFarneback(
        prev=cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY),
        next=cv2.cvtColor(current, cv2.COLOR_RGB2GRAY),
        flow=None,
        pyr_scale=float(config["pyr_scale"]),
        levels=int(config["levels"]),
        winsize=int(config["winsize"]),
        iterations=int(config["iterations"]),
        poly_n=int(config["poly_n"]),
        poly_sigma=float(config["poly_sigma"]),
        flags=int(config["flags"]),
    )
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
        raise RuntimeError(f"invalid Farneback output: {flow.shape}")
    return flow


def filter_flow_by_forward_backward_consistency(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    quality: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Zero out flow vectors that fail the in-frame cycle-consistency check."""
    cv2 = _opencv()
    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if (
        forward.ndim != 3
        or forward.shape[-1] != 2
        or backward.shape != forward.shape
        or not np.isfinite(forward).all()
        or not np.isfinite(backward).all()
    ):
        raise ValueError("forward and backward flow must be finite matching [H,W,2] arrays")

    quality = {**QUALITY_DEFAULTS, **dict(quality or {})}
    height, width = forward.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    mapped_x = grid_x + forward[..., 0]
    mapped_y = grid_y + forward[..., 1]
    in_bounds = (
        (mapped_x >= 0.0)
        & (mapped_x <= float(width - 1))
        & (mapped_y >= 0.0)
        & (mapped_y <= float(height - 1))
    )
    backward_x = cv2.remap(
        backward[..., 0], mapped_x, mapped_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    backward_y = cv2.remap(
        backward[..., 1], mapped_x, mapped_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    cycle_error_squared = (forward[..., 0] + backward_x) ** 2 + (forward[..., 1] + backward_y) ** 2
    reference_squared = (
        float(quality["relative_error"])
        * (np.square(forward).sum(axis=-1) + np.square(backward_x) + np.square(backward_y))
        + float(quality["absolute_error_squared"])
    )
    reliable = in_bounds & (cycle_error_squared <= reference_squared)
    reliable_fraction = float(reliable.mean())
    quality_valid = reliable_fraction >= float(quality["minimum_reliable_fraction"])
    filtered = np.where(reliable[..., None], forward, 0.0).astype(np.float32, copy=False)
    if not quality_valid:
        filtered.fill(0.0)
    return filtered, reliable, reliable_fraction, quality_valid


def flow_to_rgb(flow_xy: np.ndarray, *, normalization_percentile: float = 99.0) -> np.ndarray:
    """Encode direction (hue) and within-map relative magnitude (value) as RGB."""
    cv2 = _opencv()
    percentile = float(normalization_percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"normalization_percentile must be in (0, 100], got {percentile}")
    magnitude, angle = cv2.cartToPolar(flow_xy[..., 0], flow_xy[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow_xy.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 0.5, 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    scale = float(np.percentile(magnitude, percentile))
    if scale >= 1e-6:
        hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def displacement_statistics(flow_xy: np.ndarray, *, magnitude_percentile: float = 99.0) -> np.ndarray:
    """Signed mean xy, mean magnitude, and p99 magnitude. Returns (4,) float32."""
    flow_xy = np.asarray(flow_xy, dtype=np.float32)
    if flow_xy.ndim != 3 or flow_xy.shape[-1] != 2:
        raise ValueError(f"flow_xy must be [H,W,2], got {flow_xy.shape}")
    if not np.isfinite(flow_xy).all():
        raise ValueError("flow_xy contains non-finite values")
    magnitude = np.linalg.norm(flow_xy, axis=-1)
    return np.asarray(
        (
            float(flow_xy[..., 0].mean()),
            float(flow_xy[..., 1].mean()),
            float(magnitude.mean()),
            float(np.percentile(magnitude, float(magnitude_percentile))),
        ),
        dtype=np.float32,
    )


def compute_flow_observation(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    compute_size: Tuple[int, int],
    normalization_percentile: float = 99.0,
    farneback: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Return (flow_rgb, displacement_stats, reliable_fraction, quality_valid)."""
    forward_flow = compute_dense_flow(
        previous_rgb, current_rgb, compute_size=compute_size, farneback=farneback,
    )
    backward_flow = compute_dense_flow(
        current_rgb, previous_rgb, compute_size=compute_size, farneback=farneback,
    )
    flow_xy, _reliable, reliable_fraction, quality_valid = filter_flow_by_forward_backward_consistency(
        forward_flow, backward_flow, quality=quality,
    )
    return (
        flow_to_rgb(flow_xy, normalization_percentile=normalization_percentile),
        displacement_statistics(flow_xy, magnitude_percentile=normalization_percentile),
        reliable_fraction,
        quality_valid,
    )


def build_motion_features(
    displacement_stats: np.ndarray,
    interval_start_times: np.ndarray,
    interval_end_times: np.ndarray,
    interval_valid_mask: np.ndarray,
    *,
    previous_interval_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 12-D kinematic descriptors plus validity masks.

    Returns:
        features: (N, 12) float32
        interval_valid: (N,) bool
        acceleration_valid: (N,) bool
    """
    stats = np.asarray(displacement_stats, dtype=np.float64)
    starts = np.asarray(interval_start_times, dtype=np.float64)
    ends = np.asarray(interval_end_times, dtype=np.float64)
    interval_valid = np.asarray(interval_valid_mask, dtype=np.bool_)
    if stats.ndim != 2 or stats.shape[1] != 4:
        raise ValueError(f"displacement_stats must be [N,4], got {stats.shape}")
    count = int(stats.shape[0])
    if starts.shape != (count,) or ends.shape != (count,) or interval_valid.shape != (count,):
        raise ValueError("all motion interval arrays must have the same length")

    dt = ends - starts
    interval_valid = interval_valid & (dt > 0.0)
    features = np.zeros((count, MOTION_FEATURE_DIM), dtype=np.float32)
    acceleration_valid = np.zeros(count, dtype=np.bool_)
    velocity = np.zeros((count, 4), dtype=np.float64)
    velocity[interval_valid] = stats[interval_valid] / dt[interval_valid, None]
    features[interval_valid, :4] = stats[interval_valid].astype(np.float32)
    features[interval_valid, 4] = dt[interval_valid].astype(np.float32)
    features[interval_valid, 5:9] = velocity[interval_valid].astype(np.float32)

    if previous_interval_indices is None:
        previous = np.arange(count, dtype=np.int64) - 1
    else:
        previous = np.asarray(previous_interval_indices, dtype=np.int64)
        if previous.shape != (count,):
            raise ValueError("previous_interval_indices must match the interval count")

    centers = (starts + ends) * 0.5
    for index, previous_index in enumerate(previous.tolist()):
        if previous_index < 0 or previous_index >= count:
            continue
        center_dt = centers[index] - centers[previous_index]
        if (
            not interval_valid[index]
            or not interval_valid[previous_index]
            or center_dt <= 0.0
        ):
            continue
        acceleration_xy = (velocity[index, :2] - velocity[previous_index, :2]) / center_dt
        features[index, 9:11] = acceleration_xy.astype(np.float32)
        features[index, 11] = np.float32(np.linalg.norm(acceleration_xy))
        acceleration_valid[index] = True
    return features, interval_valid, acceleration_valid


@dataclass
class MotionObservation:
    flow_rgb: np.ndarray              # (K, H, W, 3) uint8
    motion_features: np.ndarray       # (K, 12) float32
    interval_valid: np.ndarray        # (K,) bool
    acceleration_valid: np.ndarray    # (K,) bool


def compute_history_motion(
    endpoint_rgbs: Sequence[np.ndarray],
    endpoint_times: Sequence[float],
    *,
    history_count: int,
    stride: int,
    compute_size: Tuple[int, int],
    normalization_percentile: float = 99.0,
    farneback: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    endpoint_indices: Optional[Sequence[int]] = None,
) -> MotionObservation:
    """Compute K history-flow frames and kinematic descriptors from K+1 endpoints.

    endpoint_rgbs / endpoint_times are ordered oldest -> newest (length K+1).
    An interval is temporally valid only when the endpoint index gap equals
    ``stride`` (if indices are given) and the timestamps increase.
    """
    k = int(history_count)
    if len(endpoint_rgbs) != k + 1 or len(endpoint_times) != k + 1:
        raise ValueError(f"expected {k + 1} endpoints, got {len(endpoint_rgbs)} / {len(endpoint_times)}")

    height, width = (int(v) for v in compute_size)
    flow_rgb = np.zeros((k, height, width, 3), dtype=np.uint8)
    displacement = np.zeros((k, 4), dtype=np.float32)
    starts = np.asarray(endpoint_times[:-1], dtype=np.float64)
    ends = np.asarray(endpoint_times[1:], dtype=np.float64)
    interval_valid = np.zeros(k, dtype=np.bool_)

    if endpoint_indices is None:
        index_ok = [True] * k
    else:
        if len(endpoint_indices) != k + 1:
            raise ValueError("endpoint_indices must have length history_count + 1")
        idxs = [int(v) for v in endpoint_indices]
        index_ok = [
            idxs[i] >= 0
            and idxs[i + 1] >= 0
            and (idxs[i + 1] - idxs[i]) == int(stride)
            for i in range(k)
        ]

    for i in range(k):
        temporal_ok = bool(index_ok[i]) and float(ends[i]) > float(starts[i])
        if not temporal_ok:
            continue
        rgb, stats, _frac, quality_valid = compute_flow_observation(
            np.asarray(endpoint_rgbs[i]),
            np.asarray(endpoint_rgbs[i + 1]),
            compute_size=compute_size,
            normalization_percentile=normalization_percentile,
            farneback=farneback,
            quality=quality,
        )
        interval_valid[i] = bool(quality_valid)
        if quality_valid:
            flow_rgb[i] = rgb
            displacement[i] = stats

    features, interval_valid, acceleration_valid = build_motion_features(
        displacement, starts, ends, interval_valid,
    )
    return MotionObservation(
        flow_rgb=flow_rgb,
        motion_features=features,
        interval_valid=interval_valid,
        acceleration_valid=acceleration_valid,
    )


def empty_motion_observation(history_count: int, compute_size: Tuple[int, int]) -> MotionObservation:
    height, width = (int(v) for v in compute_size)
    k = int(history_count)
    return MotionObservation(
        flow_rgb=np.zeros((k, height, width, 3), dtype=np.uint8),
        motion_features=np.zeros((k, MOTION_FEATURE_DIM), dtype=np.float32),
        interval_valid=np.zeros(k, dtype=np.bool_),
        acceleration_valid=np.zeros(k, dtype=np.bool_),
    )


def load_motion_stats(path: str) -> Dict[str, Any]:
    """Load DynamicWAM-style motion_stats.json (mean / scale)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload.get("scale", payload.get("standard_deviation")), dtype=np.float64)
    if mean.shape != (MOTION_FEATURE_DIM,) or scale.shape != (MOTION_FEATURE_DIM,):
        raise ValueError(f"motion stats must have {MOTION_FEATURE_DIM} values, got {mean.shape} / {scale.shape}")
    scale = np.maximum(scale, 1e-6)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "scale": scale.astype(np.float32).tolist(),
        "feature_names": list(payload.get("feature_names", MOTION_FEATURE_NAMES)),
    }


def try_load_flow_cache(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Best-effort load of a DynamicWAM exact-time flow cache (.npz)."""
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = (
                "flow_rgb",
                "motion_features",
                "interval_valid",
                "acceleration_valid",
            )
            if any(key not in payload.files for key in required):
                return None
            arrays = {key: np.asarray(payload[key]) for key in required}
            if "params" in payload.files:
                try:
                    arrays["params"] = json.loads(str(payload["params"].item()))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
    except (OSError, ValueError, KeyError):
        return None

    flow_rgb = arrays["flow_rgb"]
    features = arrays["motion_features"]
    if (
        flow_rgb.ndim != 4
        or flow_rgb.shape[-1] != 3
        or flow_rgb.dtype != np.uint8
        or features.ndim != 2
        or features.shape[0] != flow_rgb.shape[0]
        or features.shape[1] != MOTION_FEATURE_DIM
    ):
        return None
    return arrays


def slice_cache_at_anchor(
    cache: Dict[str, np.ndarray],
    anchor_idx: int,
    *,
    history_count: int,
    stride: int,
) -> Optional[MotionObservation]:
    """Take the K interval-end frames that DynamicWAM stores at ``t, t-Δ, ...``."""
    frame_count = int(cache["flow_rgb"].shape[0])
    if frame_count <= 0:
        return None
    ends = history_end_indices(anchor_idx, history_count, stride)
    if any(idx >= frame_count for idx in ends):
        return None
    return MotionObservation(
        flow_rgb=np.ascontiguousarray(cache["flow_rgb"][ends]),
        motion_features=np.ascontiguousarray(cache["motion_features"][ends]),
        interval_valid=np.ascontiguousarray(cache["interval_valid"][ends].astype(np.bool_)),
        acceleration_valid=np.ascontiguousarray(cache["acceleration_valid"][ends].astype(np.bool_)),
    )
