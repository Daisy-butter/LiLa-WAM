# Robotwin2 Dataset Loader for VLA-DINOv3 (HDF5 Version)
# Supports Robotwin2 data with HDF5 storage and flexible indexing
import os
import random
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
from PIL import Image

from models.motion import (
    IDENTITY_MOTION_STATS,
    compute_history_motion,
    empty_motion_observation,
    history_endpoint_offsets,
    interval_stride,
    load_motion_stats,
    slice_cache_at_anchor,
    try_load_flow_cache,
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

NUM_THREADS = os.cpu_count() or 4

# ImageNet normalization (DINOv3 uses ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode(buf):
    """Decode bytes to RGB numpy array (cv2 returns BGR; data already saved as RGB-style)."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # NOTE: robotwin data does not need a BGR2RGB conversion
    return img


def _normalize_image(img_np: np.ndarray) -> np.ndarray:
    """
    HxWx3 uint8 (RGB) → 3xHxW float32, ImageNet normalized.
    """
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


def detect_dataset_layout(dataset_dir) -> str:
    """Return 'dynamicwam_raw' or 'robotwin' based on the directory tree."""
    roots = [Path(p) for p in ([dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir))]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root]
        if (root / "raw").is_dir():
            candidates.append(root / "raw")
        for cand in candidates:
            for split in ("clean", "randomized"):
                split_dir = cand / split
                if not split_dir.is_dir():
                    continue
                if any(split_dir.glob("*/*/data/*.hdf5")) or any(split_dir.glob("*/data/*.hdf5")):
                    return "dynamicwam_raw"
        for child in [d for d in root.iterdir() if d.is_dir()]:
            if (child / "demo_clean").is_dir() or (child / "demo_randomized").is_dir():
                return "robotwin"
    return "robotwin"


def resolve_raw_roots(dataset_dirs: List[Path]) -> List[Path]:
    """If the user points at a DynamicWAM data/ root, prefer its raw/ subdir."""
    resolved = []
    for root in dataset_dirs:
        raw = root / "raw"
        resolved.append(raw if raw.is_dir() else root)
    return resolved


def resolve_flow_cache_root(dataset_dirs: List[Path], explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for root in dataset_dirs:
        direct = root / "flow_cache"
        if not direct.is_dir():
            continue
        nested = list(direct.glob("*/clean")) + list(direct.glob("*/randomized"))
        if nested:
            return nested[0].parent
        if any(direct.rglob("*.flow.npz")):
            return direct
    return None


def resolve_motion_stats_path(dataset_dirs: List[Path], explicit: Optional[str] = None,
                              flow_cache_root: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = []
    if flow_cache_root is not None:
        candidates.append(flow_cache_root / "motion_stats.json")
    for root in dataset_dirs:
        candidates.append(root / "flow_cache" / "motion_stats.json")
        candidates.extend(root.glob("flow_cache/*/motion_stats.json"))
    for path in candidates:
        if path.is_file():
            return path
    return None


class RobotWinTaskDataset(data.Dataset):
    def __init__(self, dataset_dir, data_mode="clean",
                 indices_config=None, camera_names=None, image_size=(320, 240),
                 val=False, image_aug=False,
                 task_cond_dir=None,
                 use_future_feat=False, future_frame_offset=None,
                 use_motion=False, motion_config=None,
                 expected_state_dim=None):
        """
        RobotWin Dataset for DINOv3-based VLA (no language input).

        Args:
            dataset_dir: dataset root directory
            data_mode: "clean" / "randomized" / "both"
            indices_config: index config {state_indices, action_indices, camera_indices}
            camera_names: camera list (the [0] primary camera is used)
            image_size: (W, H); DINOv3 requires H/W to be multiples of patch_size (=16)
            val: whether this is the validation set
            image_aug: whether to apply color augmentation
            task_cond_dir: directory of precomputed task condition vectors (mirrors the
                           dataset task structure); None = task condition disabled
            use_future_feat: whether to also return a future frame (future-feature
                             prediction training)
            future_frame_offset: offset of the future frame relative to the current
                                 anchor; None -> = chunk_size
        """
        if indices_config is None:
            raise ValueError("indices_config is required")
        if 'state_indices' not in indices_config or 'action_indices' not in indices_config:
            raise ValueError("indices_config missing required keys")

        self.dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, str) else [Path(p) for p in dataset_dir]
        self.data_mode = data_mode
        self.all_episodes = []

        # Future frame (future-feature prediction): offset defaults to chunk_size
        self.use_future_feat = use_future_feat
        self.future_frame_offset = future_frame_offset if future_frame_offset is not None else len(indices_config['action_indices'])

        self.state_offsets = torch.tensor(indices_config['state_indices'], dtype=torch.long)
        self.action_offsets = torch.tensor(indices_config['action_indices'], dtype=torch.long)
        self.chunk_size = len(self.action_offsets)
        self.indices_config = indices_config
        self.camera_names = camera_names
        # image_size: (W, H)
        self.image_size = tuple(image_size)
        assert self.image_size[0] % 16 == 0 and self.image_size[1] % 16 == 0, \
            f"DINOv3 requires H/W to be multiples of 16, got {self.image_size}"
        self.val = val
        self.image_aug = image_aug

        # Task condition vector directory
        self.task_cond_dir = task_cond_dir
        self.task_cond_cache = {}   # task_name -> torch.Tensor(D,)
        self.use_task_cond = task_cond_dir is not None

        # Dual-path motion conditioning
        self.use_motion = bool(use_motion)
        motion_config = motion_config or {}
        self.motion_history_count = int(motion_config.get('history_count', 4))
        self.motion_policy_stride = int(motion_config.get('policy_stride', 4))
        self.motion_downsample = int(motion_config.get('global_downsample_rate', 1))
        self.motion_stride = interval_stride(self.motion_policy_stride, self.motion_downsample)
        self.motion_compute_size = tuple(motion_config.get('compute_size', (64, 64)))
        self.motion_flow_image_size = tuple(motion_config.get('flow_image_size', self.motion_compute_size))
        self.motion_fps = float(motion_config.get('container_fps', 30.0))
        self.motion_percentile = float(motion_config.get('normalization_percentile', 99.0))
        self.motion_farneback = dict(motion_config.get('farneback', {}))
        self.motion_quality = dict(motion_config.get('quality', {}))
        self.flow_cache_root = None
        self.motion_stats = dict(IDENTITY_MOTION_STATS)
        self._flow_cache_mem = {}
        self.expected_state_dim = expected_state_dim
        # Prefer DynamicWAM official flow_cache; online Farnebäck only as explicit fallback
        self.require_flow_cache = bool(motion_config.get('require_flow_cache', True))
        self.allow_online_flow = bool(motion_config.get('allow_online_flow', False))

        # ColorJitter augmentation pool (lazy torchvision import — only needed when enabled)
        self.aug_pool = []
        if self.image_aug and not self.val:
            import torchvision.transforms as T
            logger.info("Initializing Image Augmentation (Randomly picking 1-2 ops)...")
            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05),
            ]

        # Resolve motion artifacts BEFORE scanning episodes so each episode can
        # attach its flow_cache path (and missing caches can be filtered out).
        if self.use_motion:
            self.flow_cache_root = resolve_flow_cache_root(
                self.dataset_dirs, motion_config.get('flow_cache_dir'),
            )
            stats_path = resolve_motion_stats_path(
                self.dataset_dirs,
                motion_config.get('stats_path'),
                self.flow_cache_root,
            )
            if stats_path is not None:
                self.motion_stats = load_motion_stats(str(stats_path))
                logger.info(f"Loaded motion statistics from {stats_path}")
            else:
                logger.warning(
                    "No motion_stats.json found; using identity mean/scale."
                )
            if self.flow_cache_root is None and self.require_flow_cache:
                raise FileNotFoundError(
                    "model.motion.enabled=True requires a flow_cache directory "
                    "(set dataset.flow_cache_dir or place flow_cache/ under dataset_dir)."
                )
            if self.flow_cache_root is not None:
                logger.info(f"Using precomputed flow cache at {self.flow_cache_root}")
            logger.info(
                f"Motion conditioning ENABLED: K={self.motion_history_count}, "
                f"stride={self.motion_stride} "
                f"(policy_stride={self.motion_policy_stride} x downsample={self.motion_downsample}); "
                f"require_flow_cache={self.require_flow_cache}, "
                f"allow_online_flow={self.allow_online_flow}"
            )

        # 1. Scan all episodes (RoboTwin or DynamicWAM raw)
        self.layout = detect_dataset_layout(self.dataset_dirs)
        logger.info(f"Detected dataset layout: {self.layout}")
        self._load_episodes()

        if self.use_task_cond:
            self._load_task_cond_vectors()

        # 2. Build the flat index map
        logger.info("Building Index Map for dataset...")
        self._build_index_map()

        if self.use_future_feat:
            logger.info(f"Future-Feat mode ENABLED: will return future frame at offset={self.future_frame_offset}.")

    def _build_index_map(self):
        """Iterate over all episodes, collect metadata and build the flat index"""
        self.valid_indices = []
        self.episode_metadata = []

        current_offset = 0
        valid_ep_count = 0

        for ep_info in tqdm(self.all_episodes, desc="Scanning episode lengths"):
            path = ep_info['hdf5_path']
            with h5py.File(path, 'r') as f:
                length = f['joint_action']['vector'].shape[0]

            if length < 2:
                continue

            self.episode_metadata.append({
                'hdf5_path': path,
                'task_name': ep_info['task_name'],
                'length': length,
                'global_start': current_offset,
                'global_end': current_offset + length,
                'split': ep_info.get('split'),
                'episode_id': ep_info.get('episode_id', Path(path).stem),
                'flow_cache_path': ep_info.get('flow_cache_path'),
            })

            ep_start = current_offset
            ep_end = current_offset + length
            curr_indices = np.arange(ep_start, ep_end, dtype=np.int64)
            self.valid_indices.append(curr_indices)

            current_offset += length
            valid_ep_count += 1

        self.valid_indices = np.concatenate(self.valid_indices)
        self._ep_end_bounds = np.array([ep['global_end'] for ep in self.episode_metadata])

        logger.info(f"Index map built: {valid_ep_count} valid episodes, {len(self.valid_indices)} total searchable frames.")

    def _load_task_cond_vectors(self):
        """Load precomputed task condition vectors {task_cond_dir}/{task_name}/task_cond.npy"""
        cond_root = Path(self.task_cond_dir)
        if not cond_root.exists():
            raise FileNotFoundError(f"task_cond_dir not found: {cond_root}")

        # Collect all task names present in the dataset
        task_names = set(ep['task_name'] for ep in self.all_episodes)
        loaded = 0
        for tn in sorted(task_names):
            npy_path = cond_root / tn / "task_cond.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Task condition vector missing for task '{tn}': {npy_path}\n"
                    f"Please run precompute_task_cond.py first to generate it.")
            vec = np.load(npy_path).astype(np.float32)
            self.task_cond_cache[tn] = torch.from_numpy(vec)
            loaded += 1
        dim = next(iter(self.task_cond_cache.values())).shape[0]
        logger.info(f"Loaded task condition vectors for {loaded} tasks (dim={dim}) from {cond_root}")

    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        data_dir = task_path / "data"
        if not data_dir.exists():
            return []
        valid_episodes = []
        for hdf5_path in data_dir.glob("*.hdf5"):
            valid_episodes.append({
                'episode_name': hdf5_path.stem,
                'task_name': task_path.parent.name if task_path.name in ['demo_clean', 'demo_randomized'] else task_path.name,
                'hdf5_path': str(hdf5_path),
                'split': split_name,
            })
        return valid_episodes

    def _load_episodes(self):
        """Walk the given root directories and find all task folders."""
        logger.info("Scanning dataset folders for all tasks...")
        if self.layout == "dynamicwam_raw":
            self._load_episodes_dynamicwam()
        else:
            self._load_episodes_robotwin()

        if not self.all_episodes:
            raise ValueError(f"No valid episodes found in: {self.dataset_dirs}")

        logger.info(f"Successfully scanned {len(self.all_episodes)} total episode files.")

    def _load_episodes_robotwin(self):
        data_splits = ["demo_clean", "demo_randomized"] if self.data_mode == "both" else [f"demo_{self.data_mode}"]
        for root_dir in self.dataset_dirs:
            if not root_dir.exists():
                continue
            for task_dir in [d for d in root_dir.iterdir() if d.is_dir()]:
                for split in data_splits:
                    split_path = task_dir / split
                    if split_path.exists():
                        self.all_episodes.extend(self._scan_task_folder(split_path, split))

    def _load_episodes_dynamicwam(self):
        """DynamicWAM / DOMINO raw: {raw}/{split}/{task}/{config}/data/episode*.hdf5"""
        split_names = ["clean", "randomized"] if self.data_mode == "both" else [self.data_mode.replace("demo_", "")]
        raw_roots = resolve_raw_roots(self.dataset_dirs)
        flow_root = self.flow_cache_root

        skipped_no_cache = 0
        for raw_root in raw_roots:
            if not raw_root.exists():
                continue
            for split in split_names:
                split_dir = raw_root / split
                if not split_dir.is_dir():
                    split_dir = raw_root / f"demo_{split}"
                if not split_dir.is_dir():
                    continue
                for task_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
                    hdf5_paths = sorted(task_dir.glob("*/data/*.hdf5")) + sorted(task_dir.glob("data/*.hdf5"))
                    # de-duplicate while preserving order
                    seen = set()
                    unique_paths = []
                    for path in hdf5_paths:
                        key = str(path.resolve())
                        if key not in seen:
                            seen.add(key)
                            unique_paths.append(path)
                    for hdf5_path in unique_paths:
                        episode_id = hdf5_path.stem.removeprefix("episode")
                        cache_path = None
                        if flow_root is not None:
                            candidate = flow_root / split / task_dir.name / "videos" / f"{episode_id}.flow.npz"
                            if candidate.is_file():
                                cache_path = str(candidate)
                        if self.use_motion and self.require_flow_cache and cache_path is None:
                            skipped_no_cache += 1
                            continue
                        self.all_episodes.append({
                            'episode_name': hdf5_path.stem,
                            'episode_id': episode_id,
                            'task_name': task_dir.name,
                            'hdf5_path': str(hdf5_path),
                            'split': split,
                            'flow_cache_path': cache_path,
                        })
        if skipped_no_cache:
            logger.warning(
                f"Skipped {skipped_no_cache} raw episodes without matching flow_cache npz."
            )

    def _get_query_indices(self, query_idx: int, episode_len: int) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
        ep_start, ep_end = 0, episode_len
        query_indices, padding_mask = {}, {}
        keys_to_process = {
            'state': self.indices_config['state_indices'],
            'action': self.indices_config['action_indices'],
        }
        for cam_name in self.camera_names:
            keys_to_process[cam_name] = self.indices_config['camera_indices']

        for key, delta_list in keys_to_process.items():
            abs_indices = [query_idx + delta for delta in delta_list]
            query_indices[key] = [max(ep_start, min(ep_end - 1, idx)) for idx in abs_indices]

            if key == 'action':
                valid_mask = [(idx < ep_end) for idx in abs_indices]
                padding_mask[f"{key}_mask"] = torch.from_numpy(np.array(valid_mask, dtype=bool))
            else:
                padding_mask[f"{key}_mask"] = torch.ones(len(abs_indices), dtype=torch.bool)

        return query_indices, padding_mask

    def _load_hdf5_data(self, hdf5_path: str, query_indices: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        with h5py.File(hdf5_path, 'r') as root:
            # Actions
            t_idx = np.array(query_indices['action'])
            h5_idx = np.unique(t_idx)
            data_batch['action_sequence'] = torch.from_numpy(
                root['joint_action']['vector'][h5_idx][np.searchsorted(h5_idx, t_idx)]
            ).float()

            # State: for DOMINO / DynamicWAM use 14-D qpos; RoboTwin static uses 16-D endpose
            t_idx = np.array(query_indices['state'])
            h5_idx = np.unique(t_idx)
            qpos_data = root['joint_action']['vector'][h5_idx]
            state_data = None
            if (
                self.expected_state_dim is not None
                and int(self.expected_state_dim) == int(qpos_data.shape[-1])
            ):
                state_data = qpos_data
            elif 'endpose' in root and 'left_endpose' in root['endpose']:
                l_pose = root['endpose']['left_endpose'][h5_idx]
                l_grip = root['endpose']['left_gripper'][h5_idx]
                r_pose = root['endpose']['right_endpose'][h5_idx]
                r_grip = root['endpose']['right_gripper'][h5_idx]
                if l_grip.ndim == 1:
                    l_grip = l_grip[:, None]
                if r_grip.ndim == 1:
                    r_grip = r_grip[:, None]
                state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            if state_data is None:
                state_data = qpos_data
            data_batch['state'] = torch.from_numpy(
                state_data[np.searchsorted(h5_idx, t_idx)]
            ).float()

            # Cameras
            data_batch['frame'] = {}
            for cam in self.camera_names:
                t_idx = np.array(query_indices[cam])
                if cam not in root['observation']:
                    continue
                h5_idx = np.unique(t_idx)
                comp_imgs = root['observation'][cam]['rgb'][h5_idx]

                decoded = np.stack([_decode(img) for img in comp_imgs])
                final_img = np.ascontiguousarray(decoded[np.searchsorted(h5_idx, t_idx)])
                if (final_img.shape[2], final_img.shape[1]) != self.image_size:
                    final_img = np.stack([
                        cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in final_img
                    ])
                data_batch['frame'][cam] = final_img

            # Future frame (future-feature prediction): single frame, primary camera only
            if self.use_future_feat and 'future_frame' in query_indices:
                cam0 = self.camera_names[0]
                if cam0 in root['observation']:
                    t_idx_f = np.array(query_indices['future_frame'])
                    h5_idx_f = np.unique(t_idx_f)
                    comp_f = root['observation'][cam0]['rgb'][h5_idx_f]
                    decoded_f = np.stack([_decode(img) for img in comp_f])
                    fut_img = np.ascontiguousarray(decoded_f[np.searchsorted(h5_idx_f, t_idx_f)])
                    if (fut_img.shape[2], fut_img.shape[1]) != self.image_size:
                        fut_img = np.stack([
                            cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in fut_img
                        ])
                    data_batch['future_frame'] = fut_img   # (1, H, W, 3) uint8 RGB
        return data_batch

    def _get_flow_cache(self, ep_meta: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        cache_path = ep_meta.get('flow_cache_path')
        if not cache_path:
            return None
        cached = self._flow_cache_mem.get(cache_path)
        if cached is not None:
            return cached
        arrays = try_load_flow_cache(cache_path)
        if arrays is None:
            return None
        # Keep only the last opened cache in this worker
        self._flow_cache_mem = {cache_path: arrays}
        return arrays

    def _load_head_frames_and_times(
        self,
        hdf5_path: str,
        indices: List[int],
        camera_name: str,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        with h5py.File(hdf5_path, 'r') as root:
            if camera_name not in root['observation']:
                raise KeyError(f"{camera_name} missing in {hdf5_path}")
            t_idx = np.array(indices, dtype=np.int64)
            h5_idx = np.unique(t_idx)
            comp = root['observation'][camera_name]['rgb'][h5_idx]
            decoded = [_decode(img) for img in comp]
            frames = [decoded[int(np.searchsorted(h5_idx, idx))] for idx in t_idx]
            if 'interception' in root and 'sim_time_seconds' in root['interception']:
                times = np.asarray(root['interception']['sim_time_seconds'][h5_idx], dtype=np.float64)
                times = times[np.searchsorted(h5_idx, t_idx)]
            else:
                times = t_idx.astype(np.float64) / max(self.motion_fps, 1e-6)
        return frames, times

    def _compute_motion_for_anchor(self, ep_meta: Dict[str, Any], local_anchor_idx: int):
        stride = self.motion_stride
        # Official path: slice DynamicWAM flow_cache (flow_rgb + 12-D features)
        cache = self._get_flow_cache(ep_meta)
        if cache is not None:
            cache_stride = stride
            params = cache.get('params') if isinstance(cache, dict) else None
            if isinstance(params, dict) and params.get('raw_stride'):
                cache_stride = int(params['raw_stride'])
            sliced = slice_cache_at_anchor(
                cache, local_anchor_idx,
                history_count=self.motion_history_count,
                stride=cache_stride,
            )
            if sliced is not None:
                return sliced

        if not self.allow_online_flow:
            return empty_motion_observation(self.motion_history_count, self.motion_compute_size)

        # Optional fallback: recompute Farnebäck online (slow; for missing caches only)
        offsets = history_endpoint_offsets(self.motion_history_count, stride)
        raw_endpoints = [int(local_anchor_idx) - off for off in offsets]
        clamped = [max(0, min(ep_meta['length'] - 1, idx)) for idx in raw_endpoints]
        try:
            frames, times = self._load_head_frames_and_times(
                ep_meta['hdf5_path'], clamped, self.camera_names[0],
            )
        except Exception as exc:
            logger.warning(f"Failed to load motion endpoints from {ep_meta['hdf5_path']}: {exc}")
            return empty_motion_observation(self.motion_history_count, self.motion_compute_size)

        return compute_history_motion(
            frames,
            times.tolist(),
            history_count=self.motion_history_count,
            stride=stride,
            compute_size=self.motion_compute_size,
            normalization_percentile=self.motion_percentile,
            farneback=self.motion_farneback,
            quality=self.motion_quality,
            endpoint_indices=raw_endpoints,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        global_curr_idx = self.valid_indices[idx]
        ep_idx = np.searchsorted(self._ep_end_bounds, global_curr_idx, side='right')
        ep_meta = self.episode_metadata[ep_idx]

        abs_start = ep_meta['global_start']

        try:
            # Local index within the episode
            local_anchor_idx = global_curr_idx - abs_start
            total_frames = ep_meta['length']

            query_indices, padding_mask = self._get_query_indices(local_anchor_idx, total_frames)

            # Future frame query index (single frame, anchor + offset)
            if self.use_future_feat:
                future_abs_idx = local_anchor_idx + self.future_frame_offset
                query_indices['future_frame'] = [
                    max(0, min(total_frames - 1, future_abs_idx))
                ]

            data_batch = self._load_hdf5_data(ep_meta['hdf5_path'], query_indices)
            data_batch.update(padding_mask)

            # Primary camera -> ImageNet-normalized pixel_values
            primary_cam = self.camera_names[0]
            pixel_values = None
            if primary_cam in data_batch['frame']:
                imgs_np = data_batch['frame'][primary_cam]   # (T, H, W, 3) uint8 RGB, T=1 for camera_indices=[0]

                # Color augmentation (PIL-based)
                if self.aug_pool:
                    num_ops = random.choice([1, 2])
                    active_ops = random.sample(self.aug_pool, num_ops)
                    new_imgs = []
                    for img_np in imgs_np:
                        pil_img = Image.fromarray(img_np)
                        for op in active_ops:
                            pil_img = op(pil_img)
                        new_imgs.append(np.array(pil_img))
                    imgs_np = np.stack(new_imgs, axis=0)

                # Normalize
                normed = np.stack([_normalize_image(img) for img in imgs_np], axis=0)   # (T, 3, H, W)
                pixel_values = torch.from_numpy(normed).float()
                # camera_indices defaults to [0] -> T=1, squeeze directly to (3, H, W)
                if pixel_values.shape[0] == 1:
                    pixel_values = pixel_values.squeeze(0)

            result = {
                'state': data_batch['state'],
                'action_sequence': data_batch['action_sequence'],
                'pixel_values': pixel_values,
                'state_mask': data_batch['state_mask'],
                'action_mask': data_batch['action_mask'],
            }

            # Future-frame pixel_values (no color augmentation; keep the supervision target clean)
            if self.use_future_feat and 'future_frame' in data_batch:
                fut_np = data_batch['future_frame']                        # (1, H, W, 3)
                fut_normed = np.stack([_normalize_image(img) for img in fut_np], axis=0)  # (1, 3, H, W)
                future_pixel_values = torch.from_numpy(fut_normed).float()
                if future_pixel_values.shape[0] == 1:
                    future_pixel_values = future_pixel_values.squeeze(0)   # (3, H, W)
                result['future_pixel_values'] = future_pixel_values

            # Task condition vector (looked up by the episode's task_name)
            if self.use_task_cond:
                task_name = ep_meta['task_name']
                result['task_cond'] = self.task_cond_cache[task_name]   # (D,)

            # Dual-path motion: rendered history-flow RGB + 12-D kinematic descriptors
            if self.use_motion:
                motion = self._compute_motion_for_anchor(ep_meta, local_anchor_idx)
                flow_imgs = motion.flow_rgb   # (K, h, w, 3)
                if (flow_imgs.shape[2], flow_imgs.shape[1]) != self.motion_flow_image_size:
                    flow_imgs = np.stack([
                        cv2.resize(img, self.motion_flow_image_size, interpolation=cv2.INTER_LINEAR)
                        for img in flow_imgs
                    ], axis=0)
                flow_normed = np.stack([_normalize_image(img) for img in flow_imgs], axis=0)
                result['flow_pixel_values'] = torch.from_numpy(flow_normed).float()          # (K, 3, H, W)
                result['motion_features'] = torch.from_numpy(motion.motion_features).float()  # (K, 12)
                result['motion_interval_valid'] = torch.from_numpy(motion.interval_valid)
                result['motion_acceleration_valid'] = torch.from_numpy(motion.acceleration_valid)

            return result

        except Exception as e:
            logger.warning(f"Error loading idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataset(config: Any, val: bool = False):
    """
    Factory function: create a RobotWinTaskDataset instance (DINOv3 version)
    """
    from omegaconf import OmegaConf
    indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)

    image_size = tuple(OmegaConf.to_container(config.dataset.image_size, resolve=True))

    # Task condition vector directory (optional)
    task_cond_dir = None
    if config.model.get('use_task_cond', False):
        task_cond_dir = config.dataset.get('task_cond_dir', None)
        if task_cond_dir is None:
            raise ValueError("model.use_task_cond=True but dataset.task_cond_dir is not set")

    # Future feature prediction (optional): whether to fetch a future frame
    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    future_frame_offset = config.dataset.get('future_frame_offset', None)

    def _plain(value, default):
        if value is None:
            return default
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    motion_cfg = config.model.get('motion', {}) or {}
    use_motion = bool(motion_cfg.get('enabled', False))
    motion_config = None
    if use_motion:
        compute_size = tuple(_plain(motion_cfg.get('compute_size', (64, 64)), (64, 64)))
        flow_image_size = tuple(_plain(motion_cfg.get('flow_image_size', compute_size), compute_size))
        motion_config = {
            'history_count': motion_cfg.get('history_count', 4),
            'policy_stride': motion_cfg.get('policy_stride', 4),
            'global_downsample_rate': motion_cfg.get('global_downsample_rate', 1),
            'compute_size': compute_size,
            'flow_image_size': flow_image_size,
            'container_fps': motion_cfg.get('container_fps', 30.0),
            'normalization_percentile': motion_cfg.get('normalization_percentile', 99.0),
            'farneback': _plain(motion_cfg.get('farneback'), {}),
            'quality': _plain(motion_cfg.get('quality'), {}),
            'flow_cache_dir': config.dataset.get('flow_cache_dir', None),
            'stats_path': config.dataset.get('motion_stats_path', None),
            'require_flow_cache': bool(motion_cfg.get('require_flow_cache', True)),
            'allow_online_flow': bool(motion_cfg.get('allow_online_flow', False)),
        }

    params = {
        'dataset_dir': config.dataset.dataset_dir,
        'indices_config': indices_config,
        'val': val,
        'image_aug': config.dataset.image_aug and not val,
        'camera_names': list(config.dataset.camera_names),
        'data_mode': config.dataset.data_mode,
        'image_size': image_size,
        'task_cond_dir': task_cond_dir,
        'use_future_feat': use_future_feat,
        'future_frame_offset': future_frame_offset,
        'use_motion': use_motion,
        'motion_config': motion_config,
        'expected_state_dim': int(config.common.state_dim),
    }

    return RobotWinTaskDataset(**params)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    result = {}
    keys = batch[0].keys()

    for key in keys:
        val = batch[0][key]
        if isinstance(val, torch.Tensor):
            result[key] = torch.stack([sample[key] for sample in batch])
        elif val is None:
            result[key] = None
        else:
            # Skip unknown types
            result[key] = [sample[key] for sample in batch]

    return result
