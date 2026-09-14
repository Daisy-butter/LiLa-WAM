#!/usr/bin/env python3
"""LiLa-WAM training (DINOv3 + optional dual-path motion).

Single GPU:
  python train.py --config ./configs/lila_dynamic.yaml --norm_stats_path ./utils/stat-domino.json

8-GPU DDP:
  torchrun --standalone --nproc_per_node=8 train.py \\
    --config ./configs/lila_dynamic.yaml \\
    --norm_stats_path ./utils/stat-domino.json \\
    --save_dir /SSD_DISK/users/wuruihan/checkpoints_lila_dynamic
"""

import os
import sys
import torch
import logging
import argparse
import time
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader.dataset import collate_fn, create_dataset
from utils.train_utils import count_parameters
from models.model_runner import ModelFactory, VLAWrapper


def setup_distributed():
    """Return (rank, local_rank, world_size, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed(is_distributed: bool):
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


class LossLogger:
    def __init__(self, log_dir="log/loss"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"train_loss_{timestamp}.csv")
        with open(self.log_file, "w") as f:
            f.write("Epoch,Step,Global_Step,Loss\n")

    def log(self, epoch, step, global_step, loss):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(f"{epoch},{step},{global_step},{loss:.6f}\n")


def build_train_config_from_yaml(cfg):
    """Extract the training-parameter dict needed by VLAWrapper from OmegaConf cfg.training"""
    t = cfg.training
    ff = cfg.model.get("future_feat", {})
    return {
        "time_mu": t.time_mu,
        "time_sigma": t.time_sigma,
        "use_vel_weight": t.use_vel_weight,
        "vel_weight_alpha": t.vel_weight_alpha,
        "vel_weight_sigma": t.vel_weight_sigma,
        "use_future_feat": ff.get("enabled", False) if ff else False,
        "lambda_future_feat": t.get("lambda_future_feat", 0.0),
    }


def unwrap_model(module):
    return module.module if isinstance(module, DDP) else module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Training Script (DINOv3) - two-stage LR version")
    parser.add_argument("--config", type=str, default="./configs/robotwin_all.yaml",
                        help="Path to config file "
                             "(use ./configs/lila_dynamic.yaml for DOMINO + dual-path motion)")
    parser.add_argument("--norm_stats_path", type=str,
                        default="./utils/stat-500-all.json",
                        help="Path to normalization stats "
                             "(use ./utils/stat-domino.json with lila_dynamic.yaml)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from after an interruption "
                             "(restores model + optimizer + scheduler + epoch, continues the "
                             "lr schedule saved in the checkpoint)")
    parser.add_argument("--init_from", type=str,
                        default=None,
                        help="Path to checkpoint to initialize model weights from "
                             "Only model weights "
                             "are loaded; optimizer and lr scheduler start fresh from --config. "
                             "Mutually exclusive with --resume.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps (dry-run / smoke). "
                             "None = run full epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-GPU batch size override (config.training.batch_size).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override config.system.num_workers.")
    parser.add_argument("--fake_dino", action="store_true",
                        help="Use FakeDINOv3 instead of local DINOv3 weights "
                             "(dry-run only; for verifying dataloader + train loop).")
    args = parser.parse_args()

    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive: "
                         "--resume continues an interrupted run (inherits its lr schedule), "
                         "--init_from starts a new stage with a fresh lr schedule.")

    rank, local_rank, world_size, distributed = setup_distributed()

    # =========================================================================
    # 1. Environment / Config
    # =========================================================================
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if is_main(rank):
        logger.info(f"Using device: {device}, Precision: {dtype}, "
                    f"world_size={world_size}, distributed={distributed}")

    config = load_config(args.config)

    epochs = config.training.epochs
    grad_accum_steps = config.training.grad_accum_steps
    save_interval_epoch = config.training.save_interval_epoch
    # batch_size is PER GPU; global batch = batch_size * world_size * grad_accum
    batch_size = int(args.batch_size) if args.batch_size is not None else int(config.training.batch_size)
    grad_clip_norm = config.training.grad_clip_norm
    lr = config.training.learning_rate
    lr_min = config.training.lr_min
    max_steps = args.max_steps

    if lr_min > lr and is_main(rank):
        logger.warning(f"lr_min ({lr_min}) > learning_rate ({lr}): "
                       f"CosineAnnealingLR will RAISE the lr towards lr_min instead of decaying. "
                       f"Check your config.")

    if is_main(rank):
        logger.info(
            f"Epochs: {epochs} | per_gpu_batch: {batch_size} | "
            f"global_batch: {batch_size * world_size * grad_accum_steps} | "
            f"grad_accum: {grad_accum_steps} | save_every: {save_interval_epoch} ep"
            + (f" | max_steps: {max_steps}" if max_steps is not None else "")
        )
        logger.info(f"LR schedule: CosineAnnealingLR {lr:.2e} -> {lr_min:.2e}")
        logger.info(f"DINOv3 feat_layers: {list(config.model.vision_encoder.feat_layers)} | "
                    f"include_cls_register: {config.model.vision_encoder.include_cls_register}")
        logger.info(f"Vel-Weight: {'ENABLED' if config.training.use_vel_weight else 'DISABLED'}")

    # =========================================================================
    # 2. Dataset / DataLoader
    # =========================================================================
    train_dataset = create_dataset(config, val=False)

    num_workers = int(args.num_workers) if args.num_workers is not None else int(config.system.num_workers)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    ) if distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    if is_main(rank):
        logger.info(f"Dataset Size: {len(train_dataset)} | Batches per Epoch (per rank): {len(train_dataloader)}")

    # =========================================================================
    # 3. Model
    # =========================================================================
    if is_main(rank):
        logger.info(">>> Initializing VLA (DINOv3)")

    if args.fake_dino:
        vision_encoder, dino_hidden_size, num_register_tokens, patch_size = (
            ModelFactory.create_fake_vision_encoder(dtype, device)
        )
    else:
        vision_encoder, dino_hidden_size, num_register_tokens, patch_size = (
            ModelFactory.create_vision_encoder(
                config.model.vision_encoder.checkpoint_path,
                dtype, device,
            )
        )
    vision_encoder.eval()

    feat_layers = list(config.model.vision_encoder.feat_layers)
    num_dino_layers = len(feat_layers)

    use_task_cond = config.model.get("use_task_cond", False)
    task_cond_dim = dino_hidden_size if use_task_cond else None
    if is_main(rank):
        logger.info(f"Task Condition: {'ENABLED' if use_task_cond else 'DISABLED'}"
                    + (f" (dim={task_cond_dim}, dir={config.dataset.get('task_cond_dir', None)})"
                       if use_task_cond else ""))

    ff_cfg = config.model.get("future_feat", {})
    use_future_feat = ff_cfg.get("enabled", False) if ff_cfg else False
    if is_main(rank):
        logger.info(f"Future-Feat Pred: {'ENABLED' if use_future_feat else 'DISABLED'}"
                    + (f" (target_layer={ff_cfg.get('target_layer', -1)}, "
                       f"lambda={config.training.get('lambda_future_feat', 0.0)})"
                       if use_future_feat else ""))

    motion_cfg = config.model.get("motion", {}) or {}
    use_motion = bool(motion_cfg.get("enabled", False))
    if use_motion and hasattr(train_dataset, "motion_stats"):
        OmegaConf.update(config, "model.motion.feature_mean", list(train_dataset.motion_stats["mean"]), merge=False)
        OmegaConf.update(config, "model.motion.feature_scale", list(train_dataset.motion_stats["scale"]), merge=False)
        if is_main(rank):
            logger.info(
                f"Dual-path motion: ENABLED "
                f"(K={motion_cfg.get('history_count', 4)}, "
                f"flow={motion_cfg.get('history_flow', True)}, "
                f"kinematic={motion_cfg.get('kinematic_tokens', True)})"
            )

    action_model = ModelFactory.create_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=num_dino_layers,
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
    )

    action_model.to(device, dtype=dtype)
    action_model.train()

    if is_main(rank):
        count_parameters(action_model, model_name="Action Model (Trainable)")

    train_config_dict = build_train_config_from_yaml(config)

    model = VLAWrapper(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=num_register_tokens,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        train_config=train_config_dict,
        future_feat_target_layer=ff_cfg.get("target_layer", -1) if ff_cfg else -1,
        flow_feat_layers=list(motion_cfg.get("flow_feat_layers", [-1])) if use_motion else None,
    )

    # =========================================================================
    # 4. Optimizer / Scheduler  (created before DDP; params still on action_model)
    # =========================================================================
    # Load weights BEFORE wrapping DDP so state_dict keys match.
    start_epoch = 0
    global_step = 0

    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_from}")
        if is_main(rank):
            logger.info(f">>> Initializing weights from {args.init_from} "
                        f"(fresh optimizer + lr schedule from config)")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        strict_load = not use_motion
        msg = action_model.load_state_dict(ckpt["model_state_dict"], strict=strict_load)
        if is_main(rank):
            logger.info(f"Model loaded (init_from epoch={ckpt.get('epoch', '?')}, strict={strict_load}). "
                        f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            if msg.missing_keys:
                logger.info(f"Missing keys (randomly initialized): {msg.missing_keys[:12]}"
                            + (" ..." if len(msg.missing_keys) > 12 else ""))
            logger.info(f"Starting new stage: lr={lr:.2e} -> {lr_min:.2e}, epochs={epochs}")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        if is_main(rank):
            logger.info(f">>> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        msg = action_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if is_main(rank):
            logger.info(f"Model loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    if distributed:
        action_model = DDP(
            action_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        model.action_model = action_model

    optimizer = AdamW(
        unwrap_model(action_model).parameters(),
        lr=lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_dataloader),
        eta_min=lr_min,
    )

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            steps_done = ckpt["epoch"] * len(train_dataloader)
            for _ in range(steps_done):
                scheduler.step()
            if is_main(rank):
                logger.warning("Old checkpoint without scheduler_state_dict; "
                               "scheduler advanced manually (lr may drift slightly).")
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("global_step",
                               start_epoch * len(train_dataloader) // grad_accum_steps)
        if is_main(rank):
            logger.info(f"Resumed at epoch={start_epoch}, global_step={global_step}, "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_save_dir = os.path.join(args.save_dir, f"sft_{timestamp}")
    if is_main(rank):
        os.makedirs(run_save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
        logger.info(f"Checkpoints will be saved to: {run_save_dir}")
        loss_logger = LossLogger(log_dir=os.path.join(run_save_dir, "loss"))
    else:
        loss_logger = None

    if distributed:
        torch.distributed.barrier()

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        # Keep frozen DINO in eval (model.train() would otherwise flip it).
        vision_encoder.eval()

        epoch_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()

        stop_training = False
        step = -1
        for step, batch in enumerate(train_dataloader):
            with torch.amp.autocast("cuda", dtype=dtype):
                loss, info_dic = model(batch)
                loss = loss / grad_accum_steps

            loss.backward()

            current_step_loss = loss.item() * grad_accum_steps
            epoch_loss += current_step_loss

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(action_model).parameters(), max_norm=grad_clip_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main(rank) and loss_logger is not None:
                    loss_logger.log(epoch + 1, step + 1, global_step, current_step_loss)

                if is_main(rank) and (global_step <= 5 or global_step % 20 == 0):
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {info_dic['loss_mse'] * grad_accum_steps:.4f} "
                        f"LR: {current_lr:.2e} "
                    )
                    if use_future_feat:
                        log_msg += f"FutureFeat: {info_dic.get('loss_future_feat', 0.0):.4f} "
                    logger.info(log_msg)

                if max_steps is not None and global_step >= int(max_steps):
                    if is_main(rank):
                        logger.info(f"Reached --max_steps={max_steps}; stopping early (smoke/dry-run).")
                    stop_training = True
                    break

        n_steps_this_epoch = step + 1
        avg_loss = epoch_loss / max(n_steps_this_epoch, 1)
        elapsed = time.time() - start_time
        if is_main(rank):
            logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ===")

        if (not stop_training) and ((epoch + 1) % save_interval_epoch == 0 or (epoch + 1) == epochs):
            if is_main(rank):
                ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
                ckpt_path = os.path.join(run_save_dir, ckpt_name)
                save_dict = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": unwrap_model(action_model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                }
                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
            if distributed:
                torch.distributed.barrier()

        if stop_training:
            break

    if is_main(rank):
        logger.info("Training Complete.")
    cleanup_distributed(distributed)
