#!/usr/bin/env python3
"""
Train CrabSmolVLA on Crab robot pick-and-place datasets.

Usage:
  python train.py
  python train.py --config configs/train_crab_smolvla.yaml --batch-size 16
"""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(Path(config_path).parent.parent)
    return cfg


def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    return torch.optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg):
    train_cfg = cfg["training"]
    warmup_steps = train_cfg["warmup_steps"]
    total_steps = train_cfg["steps"]
    min_lr = train_cfg["min_lr"]
    base_lr = train_cfg["learning_rate"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, train_loss, val_loss, output_dir, is_best=False):
    try:
        ckpt_dir = output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump({"step": step, "train_loss": train_loss, "val_loss": val_loss,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        if is_best:
            best_dir = output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            logger.info(f"New best checkpoint at step {step} (val_loss={val_loss:.4f})")
        logger.info(f"Saved checkpoint to {ckpt_dir}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {step}: {e}")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        logger.info("Continuing training despite save failure...")


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        loss = output["loss"] if isinstance(output, dict) else output
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser(description="Train CrabSmolVLA")
    parser.add_argument("--config", "-c", type=str,
                        default=str(SCRIPT_DIR / "configs" / "train_crab_smolvla.yaml"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.steps:
        cfg["training"]["steps"] = args.steps

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # Build model
    from training.crab_smolvla_wrapper import build_model
    logger.info("Building model...")
    model = build_model(cfg)

    # Patch SmolVLA config for our dims
    max_action_dim = model_cfg.get("max_action_dim", 32)
    max_state_dim = model_cfg.get("max_state_dim", 32)
    model.smolvla.config.output_features["action"].shape = (max_action_dim,)
    model.smolvla.config.input_features["observation.state"].shape = (max_state_dim,)
    logger.info(f"Patched SmolVLA config: action=({max_action_dim},), state=({max_state_dim},)")

    model.to(device)
    model.train()

    # Build dataloaders
    from training.crab_dataset import build_dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Output dir
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        logger.info(f"Resuming from {resume_dir}")
        model.load_state_dict(torch.load(resume_dir / "model.pt", map_location=device, weights_only=False))
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device, weights_only=False))
        scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location=device, weights_only=False))
        with open(resume_dir / "metadata.json") as f:
            start_step = json.load(f)["step"]
        logger.info(f"Resumed from step {start_step}")

    # Training
    total_steps = train_cfg["steps"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    log_every = train_cfg["log_every_steps"]
    eval_every = train_cfg["eval_every_steps"]
    save_every = train_cfg["checkpoint"]["save_every_steps"]
    best_val_loss = float("inf")

    logger.info(f"Training for {total_steps} steps (batch={train_cfg['batch_size']}, grad_accum={grad_accum}, effective={train_cfg['batch_size']*grad_accum})")

    step = start_step
    running_loss = 0.0
    loss_count = 0
    t_start = time.perf_counter()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            loss = output["loss"] if isinstance(output, dict) else output

            (loss / grad_accum).backward()
            running_loss += loss.item()
            loss_count += 1

            if loss_count % grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg["grad_clip_norm"]
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % log_every == 0:
                    avg_loss = running_loss / loss_count
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.perf_counter() - t_start
                    sps = (step - start_step) / max(1, elapsed)
                    eta = (total_steps - step) / max(0.01, sps)
                    logger.info(
                        f"step {step}/{total_steps} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | {sps:.2f} steps/s | ETA: {eta/60:.0f}min"
                    )
                    running_loss = 0.0
                    loss_count = 0

                if step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    logger.info(f"step {step} | val_loss={val_loss:.4f}")
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    if step % save_every == 0 or is_best:
                        save_checkpoint(model, optimizer, scheduler, step,
                                        running_loss / max(1, loss_count), val_loss, output_dir, is_best)

                elif step % save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, step,
                                    running_loss / max(1, loss_count), best_val_loss, output_dir)

    # Final save
    val_loss = evaluate(model, val_loader, device)
    logger.info(f"Final val_loss={val_loss:.4f}")
    save_checkpoint(model, optimizer, scheduler, step,
                    0.0, val_loss, output_dir, is_best=(val_loss < best_val_loss))
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
