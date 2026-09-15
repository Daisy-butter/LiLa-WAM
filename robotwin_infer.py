import os
import torch
import numpy as np
import logging
from typing import Dict, Any, List
from omegaconf import OmegaConf
from collections import deque

from models.model_runner import ModelFactory, VLAWrapper
from models.motion import (
    compute_history_motion,
    empty_motion_observation,
    history_endpoint_offsets,
    interval_stride,
    load_motion_stats,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """B-Spline smoothing of an action sequence; input and output share the same shape (N, D)"""
    from scipy.interpolate import make_lsq_spline

    N, D = action_seq.shape
    if N <= num_ctrl_pts:
        return action_seq

    x = np.arange(N)
    num_internal_knots = num_ctrl_pts - degree
    internal_knots = np.linspace(0, N - 1, num_internal_knots + 2)[1:-1]
    knots = np.concatenate([
        [0] * (degree + 1),
        internal_knots,
        [N - 1] * (degree + 1),
    ])
    spline = make_lsq_spline(x, action_seq, knots, k=degree)
    return spline(x)


def normalize_image_np(img_np: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB → 3xHxW float32 ImageNet normalized."""
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return img


def extract_joint_qpos(obs: Dict[str, Any]) -> np.ndarray:
    """14-D joint vector, matching DynamicWAM ``extract_state``."""
    if "joint_action" in obs:
        ja = obs["joint_action"]
        if isinstance(ja, dict):
            if "vector" not in ja:
                raise KeyError("observation['joint_action'] has no 'vector'")
            return np.asarray(ja["vector"], dtype=np.float32).reshape(-1)
        return np.asarray(ja, dtype=np.float32).reshape(-1)
    for key in ("qpos", "state"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1)
    raise KeyError("observation has neither joint_action/vector nor qpos")


def extract_endpose_state(obs: Dict[str, Any]) -> np.ndarray:
    """16-D endpose+gripper state used by original LiLa-WAM / RoboTwin static."""
    if "endpose" not in obs:
        raise KeyError("observation has no 'endpose'")
    endpose = obs["endpose"]
    l_pose = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
    l_grip = np.asarray([endpose["left_gripper"]], dtype=np.float32)
    r_pose = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1)
    r_grip = np.asarray([endpose["right_gripper"]], dtype=np.float32)
    return np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=0)


_SCENE_CLOCK_ERROR = (
    "absolute-motion eval requires TASK_ENV._scene_step_clock "
    "(DynamicWAM DOMINO SceneStepClock patch). Without it, kinematics dt would "
    "be silently wrong. Apply third_party/domino/evaluated.patch before eval; "
    "fps/container_fps fallback is not allowed."
)


def require_scene_step_clock(task_env: Any) -> Any:
    """Hard-fail if the live env has no exact DynamicWAM simulator clock."""
    if task_env is None:
        raise RuntimeError(_SCENE_CLOCK_ERROR)
    scene_clock = getattr(task_env, "_scene_step_clock", None)
    if scene_clock is None:
        raise RuntimeError(_SCENE_CLOCK_ERROR)
    if hasattr(scene_clock, "installed") and not bool(scene_clock.installed):
        raise RuntimeError(
            _SCENE_CLOCK_ERROR + " Clock exists but is not installed on scene.step()."
        )
    if not hasattr(scene_clock, "snapshot"):
        raise RuntimeError(_SCENE_CLOCK_ERROR + " Clock is missing snapshot().")
    snapshot = scene_clock.snapshot()
    timestamp = getattr(snapshot, "time_seconds", None)
    if timestamp is None or not np.isfinite(float(timestamp)):
        raise RuntimeError(
            _SCENE_CLOCK_ERROR + " snapshot().time_seconds is missing or non-finite."
        )
    return scene_clock


def _finite_timestamp(raw: Any) -> float:
    if raw is None:
        raise ValueError("timestamp is None")
    value = np.asarray(raw, dtype=np.float64).reshape(-1)
    if value.size == 0:
        raise ValueError("timestamp is empty")
    timestamp = float(value[-1])
    if not np.isfinite(timestamp):
        raise ValueError(f"timestamp is non-finite: {timestamp}")
    return timestamp


def resolve_simulator_time_seconds(
    observation: Dict[str, Any],
    task_env: Any = None,
) -> float:
    """Exact simulator time, matching DynamicWAM absolute-motion deploy.

    Live eval (task_env bound): ALWAYS read TASK_ENV._scene_step_clock.
    Replay / unit tests (no env): observation['sim_time_seconds'] or interception.
    """
    if task_env is not None:
        snapshot = require_scene_step_clock(task_env).snapshot()
        return _finite_timestamp(snapshot.time_seconds)

    candidates: List[Any] = [observation.get("sim_time_seconds", None)]
    interception = observation.get("interception")
    if isinstance(interception, dict):
        candidates.append(interception.get("sim_time_seconds", None))
    for raw in candidates:
        if raw is None:
            continue
        try:
            return _finite_timestamp(raw)
        except ValueError:
            continue
    raise RuntimeError(
        "absolute-motion inference requires exact simulator time "
        "(bind_env(TASK_ENV) with _scene_step_clock, or set "
        "observation['sim_time_seconds']); fps fallback is not allowed"
    )


class RobotWinInference:
    """
    RobotWin VLA inference class (DINOv3 version, no language input)
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        norm_stats_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        task_name: str = None,
    ):
        self.device = device
        self.dtype = dtype

        # 1. Config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = OmegaConf.load(config_path)

        self.num_inference_steps = self.config.common.num_inference_steps
        self.action_execution_horizon = self.config.common.action_execution_horizon

        # Smoothing config
        self.smooth_actions = self.config.inference.smooth_actions
        self.smooth_sigma = self.config.inference.smooth_sigma

        time_sampler = self.config.training.time_sampler

        # Image size
        self.image_size = tuple(OmegaConf.to_container(self.config.dataset.image_size, resolve=True))   # (W, H)

        # Action queue
        self.action_queue = deque()

        # 2. Models
        logger.info(f"Loading Model from {checkpoint_path}...")

        vision_encoder, dino_hidden_size, num_register_tokens, patch_size = ModelFactory.create_vision_encoder(
            self.config.model.vision_encoder.checkpoint_path,
            dtype, device,
        )

        feat_layers = list(self.config.model.vision_encoder.feat_layers)
        num_dino_layers = len(feat_layers)

        # Task condition vector config
        self.use_task_cond = self.config.model.get('use_task_cond', False)
        self.task_cond_dim = dino_hidden_size if self.use_task_cond else None
        self.task_cond_dir = self.config.dataset.get('task_cond_dir', None) if self.use_task_cond else None
        self.task_cond = None   # condition vector of the current task (B=1, D), set via set_task()

        motion_cfg = self.config.model.get('motion', {}) or {}
        self.use_motion = bool(motion_cfg.get('enabled', False))
        if self.use_motion:
            stats_path = self.config.dataset.get('motion_stats_path', None)
            if not stats_path:
                raise FileNotFoundError(
                    "motion.enabled=True but dataset.motion_stats_path is not set"
                )
            stats = load_motion_stats(str(stats_path))
            OmegaConf.update(self.config, "model.motion.feature_mean", stats["mean"], merge=False)
            OmegaConf.update(self.config, "model.motion.feature_scale", stats["scale"], merge=False)

        action_model = ModelFactory.create_action_model(
            self.config,
            dino_hidden_size=dino_hidden_size,
            num_dino_layers=num_dino_layers,
            task_cond_dim=self.task_cond_dim,
            patch_size=patch_size,
        )

        ff_cfg = self.config.model.get('future_feat', {})
        self.model = VLAWrapper(
            vision_encoder=vision_encoder,
            action_model=action_model,
            time_sampler=time_sampler,
            feat_layers=feat_layers,
            include_cls_register=self.config.model.vision_encoder.include_cls_register,
            num_register_tokens=num_register_tokens,
            device=device,
            dtype=dtype,
            norm_stats_path=norm_stats_path,
            train_config=None,
            future_feat_target_layer=ff_cfg.get('target_layer', -1) if ff_cfg else -1,
            flow_feat_layers=list(motion_cfg.get('flow_feat_layers', [-1])) if self.use_motion else None,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        msg = self.model.action_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded Action Model weights. Missing: {len(msg.missing_keys)}, "
                    f"Unexpected: {len(msg.unexpected_keys)}")

        self.model.eval()
        self.model.to(device, dtype)

        self.expected_state_dim = int(self.config.common.state_dim)
        self.expected_action_dim = int(self.config.common.action_dim)
        stats_state_dim = int(self.model.state_min.numel())
        stats_action_dim = int(self.model.action_min.numel())
        if stats_state_dim != self.expected_state_dim:
            raise ValueError(
                f"norm-stats state dim {stats_state_dim} != config.state_dim "
                f"{self.expected_state_dim}"
            )
        if stats_action_dim != self.expected_action_dim:
            raise ValueError(
                f"norm-stats action dim {stats_action_dim} != config.action_dim "
                f"{self.expected_action_dim}"
            )
        self.task_env = None

        # Processor
        indices_config = self.config.dataset.indices_config
        self.processor = RobotWinInferenceProcessor(
            indices_config=indices_config,
            camera_name=list(self.config.dataset.camera_names)[0],
            image_size=self.image_size,
            device=device,
            dtype=dtype,
            motion_config=motion_cfg if self.use_motion else None,
            expected_state_dim=self.expected_state_dim,
        )

        # If task_name is given at construction time, load it immediately
        if self.use_task_cond and task_name is not None:
            self.set_task(task_name)

        logger.info(f"Inference Engine Ready. "
                    f"Smooth: {self.smooth_actions}; "
                    f"feat_layers={feat_layers}, image_size={self.image_size}; "
                    f"TaskCond: {self.use_task_cond}; "
                    f"Motion: {self.use_motion}; "
                    f"state_dim: {self.expected_state_dim}")

    def bind_env(self, task_env: Any):
        """Bind the live RoboTwin/DOMINO env for exact simulator-time motion."""
        self.task_env = task_env
        if self.use_motion:
            require_scene_step_clock(task_env)
            logger.info(
                "Bound TASK_ENV._scene_step_clock for absolute-motion kinematics "
                f"(t={float(task_env._scene_step_clock.snapshot().time_seconds):.4f}s)"
            )

    def set_task(self, task_name: str):
        """Load the precomputed condition vector of the given task.
        Call once before each task starts during evaluation."""
        if not self.use_task_cond:
            return
        if self.task_cond_dir is None:
            raise ValueError("use_task_cond=True but dataset.task_cond_dir is not configured")
        npy_path = os.path.join(self.task_cond_dir, task_name, "task_cond.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Task condition vector not found for '{task_name}': {npy_path}")
        vec = np.load(npy_path).astype(np.float32)
        self.task_cond = torch.from_numpy(vec).to(self.device, self.dtype).unsqueeze(0)  # (1, D)
        logger.info(f"Loaded task condition for '{task_name}' (dim={vec.shape[0]}, |v|={np.linalg.norm(vec):.3f})")

    def reset(self):
        """Reset state: clear the Processor history buffer and the action queue"""
        self.processor.reset()
        self.action_queue.clear()

    def _observation_with_simulator_time(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_motion:
            return observation
        if self.task_env is None:
            raise RuntimeError(
                _SCENE_CLOCK_ERROR + " Call bind_env(TASK_ENV) before model.step()."
            )
        timestamp = resolve_simulator_time_seconds(observation, self.task_env)
        enriched = dict(observation)
        enriched["sim_time_seconds"] = timestamp
        return enriched

    @torch.no_grad()
    def _predict_chunk(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """[Internal] Run one model inference to generate an action chunk.
        instruction is a placeholder only and is not used."""
        if self.use_task_cond and self.task_cond is None:
            raise RuntimeError("use_task_cond=True but set_task(task_name) was not called "
                               "to set the task condition vector")
        # 1. Preprocess
        batch = self.processor.process(observation)

        # 2. Conditioning
        qpos_cond = self.model.normalize_state(batch['state'])
        dino_features_list = self.model.get_vision_features(batch['pixel_values'])

        # 3. Flow Matching sampling preparation
        B = 1
        action_len = self.config.common.action_chunk_size
        action_dim = self.config.common.action_dim

        x_t = torch.randn((B, action_len, action_dim), device=self.device, dtype=self.dtype)
        steps = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device, dtype=self.dtype)

        # 4. ODE Solver
        for i in range(self.num_inference_steps):
            t_curr = steps[i]
            dt = steps[i+1] - t_curr
            t_input = t_curr.unsqueeze(0)

            motion_kwargs = {}
            if self.use_motion:
                motion_kwargs = {
                    "motion_features": batch.get("motion_features"),
                    "motion_interval_valid": batch.get("motion_interval_valid"),
                    "motion_acceleration_valid": batch.get("motion_acceleration_valid"),
                    "flow_features": None,
                }
                if batch.get("flow_pixel_values") is not None:
                    flow_pv = batch["flow_pixel_values"]
                    Bf, K = flow_pv.shape[:2]
                    flow_flat = flow_pv.reshape(Bf * K, *flow_pv.shape[2:])
                    flow_layer_feats = self.model.get_vision_features(
                        flow_flat, feat_layers=self.model.flow_feat_layers,
                    )
                    motion_kwargs["flow_features"] = flow_layer_feats[-1].reshape(
                        Bf, K, *flow_layer_feats[-1].shape[1:]
                    )

            preds = self.model.action_model(
                t=t_input,
                noisy_actions=x_t,
                qpos_history=qpos_cond,
                dino_features_list=dino_features_list,
                task_cond=self.task_cond,
                **motion_kwargs,
            )

            pred_v = preds["final_pred"]

            x_t = x_t + pred_v * dt

        # 5. Denormalize
        action_seq = self.model.denormalize_action(x_t)
        action_np = action_seq[0].float().cpu().numpy()

        if self.smooth_actions:
            action_np = bspline_smooth(action_np, degree=3, num_ctrl_pts=8)

        return action_np

    def step(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """
        [Public interface] Receding Horizon Control
        - Queue empty -> run inference for a new chunk, enqueue the first N actions
        - Queue non-empty -> pop the front action

        The instruction parameter is kept for compatibility with legacy callers;
        it is not used internally (the DINOv3 version has no language input).
        """
        observation = self._observation_with_simulator_time(observation)

        # Always update the lightweight state buffer so the proprioception
        # history stays continuous
        self.processor.update_state_buffer(observation)

        if len(self.action_queue) == 0:
            full_chunk = self._predict_chunk(observation, instruction)
            valid_actions = full_chunk[:self.action_execution_horizon]
            for act in valid_actions:
                self.action_queue.append(act)

        return self.action_queue.popleft()


class RobotWinInferenceProcessor:
    """
    Real-time inference preprocessing (DINOv3 version, no VLM):
    1. Maintains the Proprioception (State) history buffer
    2. Converts environment observations into a pixel_values Tensor (ImageNet normalized)
    3. Optional Dual-path motion (online Farnebäck + exact simulator time)
    """
    def __init__(
        self,
        indices_config: Dict[str, List[int]] = None,
        camera_name: str = 'head_camera',
        image_size=(320, 240),    # (W, H)
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        motion_config: Dict[str, Any] = None,
        expected_state_dim: int = 16,
    ):
        self.device = device
        self.dtype = dtype
        self.camera_name = camera_name
        self.image_size = tuple(image_size)
        self.expected_state_dim = int(expected_state_dim)

        self.state_indices = indices_config['state_indices']
        self.history_len = 1 + abs(min(self.state_indices))

        self.state_buffer = deque(maxlen=self.history_len)

        self.motion_config = motion_config
        self.use_motion = motion_config is not None and bool(motion_config.get('enabled', True))
        if self.use_motion:
            self.motion_history_count = int(motion_config.get('history_count', 4))
            self.motion_stride = interval_stride(
                int(motion_config.get('policy_stride', 4)),
                int(motion_config.get('global_downsample_rate', 1)),
            )
            self.motion_compute_size = tuple(motion_config.get('compute_size', (64, 64)))
            self.motion_flow_image_size = tuple(
                motion_config.get('flow_image_size', self.motion_compute_size)
            )
            # container_fps is cache metadata only; never used as a kinematics clock.
            self.frame_buffer = deque(maxlen=self.motion_history_count * self.motion_stride + 1)
            self.time_buffer = deque(maxlen=self.motion_history_count * self.motion_stride + 1)
            self.frame_index = 0
        else:
            self.frame_buffer = None
            self.time_buffer = None
            self.frame_index = 0

    def reset(self):
        self.state_buffer.clear()
        if self.frame_buffer is not None:
            self.frame_buffer.clear()
            self.time_buffer.clear()
        self.frame_index = 0

    def update_state_buffer(self, observation: Dict[str, Any]):
        """Update proprioception and (if enabled) the head-camera motion history."""
        current_state = self._parse_state_from_obs(observation)
        if len(self.state_buffer) == 0:
            for _ in range(self.history_len):
                self.state_buffer.append(current_state)
        else:
            self.state_buffer.append(current_state)
        self._append_motion_frame(observation)

    def _append_motion_frame(self, observation: Dict[str, Any]):
        if not self.use_motion or self.frame_buffer is None:
            return
        if self.camera_name not in observation.get('observation', {}):
            raise KeyError(
                f"motion.enabled=True but camera '{self.camera_name}' is missing "
                "from observation['observation']"
            )
        import cv2
        img_np = observation['observation'][self.camera_name]['rgb']
        if (img_np.shape[1], img_np.shape[0]) != self.image_size:
            img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_LINEAR)
        # Exact simulator time only (DynamicWAM HeadFlowBuffer contract).
        timestamp = resolve_simulator_time_seconds(observation, task_env=None)
        if self.time_buffer and timestamp < self.time_buffer[-1]:
            raise ValueError(
                "simulator time cannot move backwards between policy frames: "
                f"{self.time_buffer[-1]:.9f} -> {timestamp:.9f}"
            )
        self.frame_buffer.append(np.ascontiguousarray(img_np))
        self.time_buffer.append(float(timestamp))
        self.frame_index += 1

    def _parse_state_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """Select proprio by ``expected_state_dim`` (train/serve must match).

        - 14: DynamicWAM / DOMINO joint qpos (``joint_action/vector``)
        - 16: original LiLa endpose+gripper
        """
        if self.expected_state_dim == 14:
            state = extract_joint_qpos(obs)
        elif self.expected_state_dim == 16:
            state = extract_endpose_state(obs)
        else:
            # Best-effort: prefer a source whose flattened dim matches config.
            errors = []
            for extractor in (extract_joint_qpos, extract_endpose_state):
                try:
                    candidate = extractor(obs)
                except KeyError as exc:
                    errors.append(str(exc))
                    continue
                if int(candidate.shape[-1]) == self.expected_state_dim:
                    state = candidate
                    break
            else:
                raise KeyError(
                    f"could not build state_dim={self.expected_state_dim} from observation "
                    f"({'; '.join(errors) or 'no candidates'})"
                )

        if int(state.shape[-1]) != self.expected_state_dim:
            raise ValueError(
                f"proprio dim mismatch: got {state.shape[-1]}, "
                f"expected state_dim={self.expected_state_dim}"
            )
        return state

    def process(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process an inference input; assumes update_state_buffer has been called"""
        # State
        state_seq_np = np.stack(list(self.state_buffer), axis=0)
        state_tensor = torch.from_numpy(state_seq_np).to(self.device, self.dtype).unsqueeze(0)

        # Image
        pixel_values = None
        if self.camera_name in observation['observation']:
            import cv2
            img_np = observation['observation'][self.camera_name]['rgb']    # HxWx3 RGB uint8

            # Resize to the target size (W, H)
            if (img_np.shape[1], img_np.shape[0]) != self.image_size:
                img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_LINEAR)

            normed = normalize_image_np(img_np)   # (3, H, W)
            pixel_values = torch.from_numpy(normed).to(self.device, self.dtype).unsqueeze(0)
        else:
            logger.warning(f"Camera {self.camera_name} not found in observation!")

        result = {
            'state': state_tensor,
            'pixel_values': pixel_values,
        }

        if self.use_motion:
            result.update(self._build_motion_tensors())

        return result

    def _build_motion_tensors(self) -> Dict[str, torch.Tensor]:
        import cv2
        offsets = history_endpoint_offsets(self.motion_history_count, self.motion_stride)
        n = len(self.frame_buffer)
        if n == 0:
            motion = empty_motion_observation(self.motion_history_count, self.motion_compute_size)
        else:
            newest = n - 1
            raw_endpoints = [newest - off for off in offsets]
            clamped = [max(0, idx) for idx in raw_endpoints]
            frames = [self.frame_buffer[idx] for idx in clamped]
            times = [self.time_buffer[idx] for idx in clamped]
            motion = compute_history_motion(
                frames,
                times,
                history_count=self.motion_history_count,
                stride=self.motion_stride,
                compute_size=self.motion_compute_size,
                normalization_percentile=float(self.motion_config.get('normalization_percentile', 99.0)),
                farneback=dict(self.motion_config.get('farneback', {}) or {}),
                quality=dict(self.motion_config.get('quality', {}) or {}),
                endpoint_indices=raw_endpoints,
            )

        flow_imgs = motion.flow_rgb
        target_size = self.motion_flow_image_size
        if (flow_imgs.shape[2], flow_imgs.shape[1]) != target_size:
            flow_imgs = np.stack(
                [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in flow_imgs],
                axis=0,
            )
        flow_normed = np.stack([normalize_image_np(img) for img in flow_imgs], axis=0)
        return {
            'flow_pixel_values': torch.from_numpy(flow_normed).to(self.device, self.dtype).unsqueeze(0),
            'motion_features': torch.from_numpy(motion.motion_features).to(self.device, self.dtype).unsqueeze(0),
            'motion_interval_valid': torch.from_numpy(motion.interval_valid).to(self.device).unsqueeze(0),
            'motion_acceleration_valid': torch.from_numpy(motion.acceleration_valid).to(self.device).unsqueeze(0),
        }


class ActionRecorder:
    def __init__(self):
        self.actions = []

    def record(self, action):
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        self.actions.append(action)

    def plot_and_save(self, save_dir, episode_id):
        if not self.actions:
            return

        import matplotlib.pyplot as plt

        actions_np = np.array(self.actions)
        T, D = actions_np.shape

        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        axes = axes.flatten()

        for d in range(min(D, 14)):
            axes[d].plot(actions_np[:, d], color='b')
            axes[d].set_title(f'Action Dimension {d} (Joint Angle)')
            axes[d].set_xlabel('Step')
            axes[d].set_ylabel('Value')
            axes[d].grid(True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'action_episode_{episode_id}.png')
        plt.savefig(save_path)
        plt.close(fig)
        self.actions = []


if __name__ == "__main__":
    agent = RobotWinInference(
        config_path="./configs/robotwin_all.yaml",
        checkpoint_path="./checkpoints_vla/sft_2026-07-25_22-40-05/checkpoint_epoch_3.pt",
        norm_stats_path="./utils/stat-500-all.json",
    )
    # Simulated environment loop:
    # obs = env.reset()
    # agent.reset()
    # for i in range(100):
    #     action = agent.step(obs)
    #     obs, _, _, _ = env.step(action)
