"""
CrabSmolVLA: SmolVLA wrapper adapted for Crab robot.

v3: Loss masking support for unified 14-DOF training.
    Active joints per task defined in config task_masks.
    Configurable action_dim/state_dim (supports joint subsets).
    Tactile encoder optional (disabled via config).

v4: Optional Reward-Weighted Flow Matching (RWFM) reduction.
    Computes per-sample loss and applies reward weights when enabled=true.
    Preserves baseline behavior when enabled=false.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrabSmolVLAWrapper(nn.Module):
    def __init__(self, cfg: dict, smolvla_policy, tactile_encoder):
        super().__init__()
        self.cfg = cfg
        self.smolvla = smolvla_policy
        self.tactile_encoder = tactile_encoder

        model_cfg = cfg["model"]
        self.action_dim = model_cfg.get("action_dim", 14)
        self.state_dim = model_cfg.get("state_dim", 14)
        self.max_state_dim = model_cfg.get("max_state_dim", 32)
        self.max_action_dim = model_cfg.get("max_action_dim", 32)
        self.chunk_size = model_cfg.get("chunk_size", 50)

        # State augmentation projection
        tactile_dim = self.tactile_encoder.total_output_dim if self.tactile_encoder.enabled else 0
        augmented_state_dim = self.state_dim + tactile_dim

        self.state_augment_proj = nn.Linear(augmented_state_dim, self.max_state_dim)
        nn.init.xavier_uniform_(self.state_augment_proj.weight)
        nn.init.zeros_(self.state_augment_proj.bias)

        # Camera key remapping (our dataset -> SmolVLA expected keys)
        self.camera_remap = {
            "observation.images.main_camera": "observation.images.camera1",
            "observation.images.left_arm_camera": "observation.images.camera2",
            "observation.images.right_arm_camera": "observation.images.camera3",
        }

        # Loss masking config
        self.task_masks = cfg.get("task_masks", None)
        if self.task_masks:
            logger.info(f"Loss masking enabled with {len(self.task_masks)} task masks:")
            for prefix, indices in self.task_masks.items():
                logger.info(f"  '{prefix}' -> joints {indices}")

        # Optional reward-weighted flow matching configuration
        rwfm_cfg = cfg.get("reward_weighting", {}) or {}
        self.rwfm_enabled = bool(rwfm_cfg.get("enabled", False))
        self.rwfm_alpha = float(rwfm_cfg.get("alpha", 0.5))
        self.rwfm_alpha_warmup_steps = int(rwfm_cfg.get("alpha_warmup_steps", 0))
        self.rwfm_alpha_ramp_steps = int(rwfm_cfg.get("alpha_ramp_steps", 0))
        self.rwfm_w_min = float(rwfm_cfg.get("w_min", 0.25))
        self.rwfm_w_max = float(rwfm_cfg.get("w_max", 4.0))
        self.rwfm_mix_beta = float(rwfm_cfg.get("mix_beta", 0.7))
        self.rwfm_z_clip = float(rwfm_cfg.get("z_clip", 6.0))
        self.rwfm_a_clip = float(rwfm_cfg.get("a_clip", 6.0))
        self.rwfm_group_key = str(rwfm_cfg.get("group_key", "task"))  # task | family | dataset
        self.rwfm_eps = float(rwfm_cfg.get("eps", 1e-6))
        self._rwfm_missing_reward_warned = False
        self._rwfm_missing_groups_warned = set()

        self.rwfm_group_stats = {}
        stats_path_cfg = rwfm_cfg.get("stats_path", "docs/reward_weighting_stats.json")
        config_dir = Path(cfg.get("_config_dir", "."))
        stats_path = Path(stats_path_cfg)
        if not stats_path.is_absolute():
            stats_path = config_dir / stats_path
        self.rwfm_stats_path = stats_path
        if self.rwfm_enabled:
            self._load_rwfm_group_stats()

        # Anchor regularization (anti-collapse): L_anchor = ||theta - theta_init||^2
        anchor_cfg = cfg.get("anchor", {}) or {}
        self.anchor_enabled = bool(anchor_cfg.get("enabled", False))
        self.anchor_lambda = float(anchor_cfg.get("lambda", 0.0))
        self.anchor_lambda_warmup_steps = int(anchor_cfg.get("warmup_steps", 0))
        self.anchor_lambda_ramp_steps = int(anchor_cfg.get("ramp_steps", 0))
        self.anchor_trainable_only = bool(anchor_cfg.get("trainable_only", True))
        self.anchor_reference_device = str(anchor_cfg.get("reference_device", "same"))  # same | cpu
        self.anchor_param_max = int(anchor_cfg.get("param_max", 0))  # 0 = all
        self._anchor_ref_params = {}
        self._anchor_ready = False
        self._anchor_numel = 0
        self._anchor_param_count = 0
        self._anchor_warned_no_params = False

        # Internal training step counter for alpha/lambda ramp when no external step is provided
        self._train_forward_steps = 0

        # Cache tokenizer from SmolVLA for language encoding
        self._tokenizer = None
        self._tokenizer_max_length = 48
        try:
            self._tokenizer = smolvla_policy.model.vlm_with_expert.processor.tokenizer
            if hasattr(smolvla_policy.config, "tokenizer_max_length"):
                self._tokenizer_max_length = smolvla_policy.config.tokenizer_max_length
            logger.info(f"Cached SmolVLA tokenizer, max_length={self._tokenizer_max_length}")
        except Exception as e:
            logger.warning(f"Could not cache SmolVLA tokenizer: {e}")

        logger.info(
            f"CrabSmolVLAWrapper: action_dim={self.action_dim}, state_dim={self.state_dim}, "
            f"tactile_dim={tactile_dim}, augmented={augmented_state_dim} -> {self.max_state_dim}"
        )
        if self.rwfm_enabled:
            logger.info(
                "RWFM enabled: alpha=%.3f, w_clip=[%.3f, %.3f], beta=%.3f, group=%s, stats=%s",
                self.rwfm_alpha,
                self.rwfm_w_min,
                self.rwfm_w_max,
                self.rwfm_mix_beta,
                self.rwfm_group_key,
                self.rwfm_stats_path,
            )
        if self.anchor_enabled:
            logger.info(
                "Anchor enabled: lambda=%.3e, warmup=%d, ramp=%d, trainable_only=%s, ref_device=%s",
                self.anchor_lambda,
                self.anchor_lambda_warmup_steps,
                self.anchor_lambda_ramp_steps,
                self.anchor_trainable_only,
                self.anchor_reference_device,
            )

    def _load_rwfm_group_stats(self):
        """Load robust normalization stats from Step-3 artifact if available."""
        if not self.rwfm_stats_path.exists():
            logger.warning(
                "RWFM stats file not found: %s (fallback to batch-local robust stats)",
                self.rwfm_stats_path,
            )
            return

        try:
            payload = json.loads(self.rwfm_stats_path.read_text(encoding="utf-8"))
            self.rwfm_group_stats = payload.get("group_stats", {}) or {}
            logger.info("Loaded RWFM group stats for %d groups", len(self.rwfm_group_stats))
        except Exception as e:
            logger.warning(
                "Failed loading RWFM stats from %s: %s (fallback to batch-local stats)",
                self.rwfm_stats_path,
                e,
            )
            self.rwfm_group_stats = {}

    def _resolve_training_step(self, batch: dict) -> int:
        """
        Resolve step for schedules.
        Priority:
          1) external `batch["train_step"]` if present
          2) internal forward-call counter during training
        """
        if "train_step" in batch:
            raw = batch["train_step"]
            if isinstance(raw, torch.Tensor):
                return int(raw.item())
            return int(raw)

        if self.training:
            step = self._train_forward_steps
            self._train_forward_steps += 1
            return step

        return self._train_forward_steps

    def _scheduled_scalar(self, base: float, warmup_steps: int, ramp_steps: int, step: int) -> float:
        """Linear warmup+ramp schedule for alpha/lambda."""
        if base <= 0:
            return 0.0
        if step < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return float(base)

        progress = (step - warmup_steps) / max(1, ramp_steps)
        progress = float(np.clip(progress, 0.0, 1.0))
        return float(base * progress)

    def _maybe_init_anchor_references(self):
        """Capture theta_init once (lazy init) for parameter-anchor regularization."""
        if not self.anchor_enabled or self._anchor_ready:
            return

        ref_device = self.anchor_reference_device.lower()
        if ref_device not in {"same", "cpu"}:
            logger.warning("Unknown anchor reference_device=%s, fallback to 'same'", ref_device)
            ref_device = "same"

        count = 0
        numel = 0
        for name, param in self.named_parameters():
            if self.anchor_trainable_only and not param.requires_grad:
                continue

            ref = param.detach().clone()
            if ref_device == "cpu":
                ref = ref.cpu()

            self._anchor_ref_params[name] = ref
            count += 1
            numel += ref.numel()

            if self.anchor_param_max > 0 and count >= self.anchor_param_max:
                break

        self._anchor_param_count = count
        self._anchor_numel = numel
        self._anchor_ready = True

        if count == 0:
            logger.warning("Anchor enabled but no parameters were captured for regularization")
        else:
            logger.info(
                "Anchor references initialized: %d params, %d elements (device=%s)",
                count,
                numel,
                ref_device,
            )

    def _compute_anchor_loss(self, device: torch.device) -> torch.Tensor:
        """Compute mean squared parameter drift from initialization."""
        if not self.anchor_enabled:
            return torch.zeros((), device=device, dtype=torch.float32)

        self._maybe_init_anchor_references()
        if not self._anchor_ref_params:
            if not self._anchor_warned_no_params:
                logger.warning("Anchor has no reference params; anchor loss will be zero")
                self._anchor_warned_no_params = True
            return torch.zeros((), device=device, dtype=torch.float32)

        total = torch.zeros((), device=device, dtype=torch.float32)
        denom = 0
        for name, param in self.named_parameters():
            ref = self._anchor_ref_params.get(name, None)
            if ref is None:
                continue
            if self.anchor_trainable_only and not param.requires_grad:
                continue

            if ref.device != param.device:
                ref_t = ref.to(device=param.device, dtype=param.dtype, non_blocking=True)
            else:
                ref_t = ref.to(dtype=param.dtype)

            diff = param - ref_t
            total = total + (diff * diff).sum()
            denom += diff.numel()

        if denom == 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        return total / denom

    def _robust_location_scale(self, x: np.ndarray) -> tuple[float, float]:
        """Robust median/MAD location-scale with std fallback."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return 0.0, 1.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        if scale < self.rwfm_eps:
            std = float(np.std(x))
            scale = std if std > self.rwfm_eps else 1.0
        return med, float(scale)

    def _resolve_group_names(self, batch: dict, batch_size: int) -> list[str]:
        """Resolve normalization group for each sample."""
        if self.rwfm_group_key == "task":
            names = batch.get("task_name", batch.get("task", None))
            if names is None:
                return ["unknown"] * batch_size
            return [str(x) for x in names]

        if self.rwfm_group_key == "dataset":
            names = batch.get("dataset_name", None)
            if names is None:
                return ["unknown"] * batch_size
            return [str(x) for x in names]

        if self.rwfm_group_key == "family":
            names = batch.get("dataset_name", None)
            if names is None:
                return ["unknown"] * batch_size
            out = []
            for name in names:
                text = str(name)
                out.append(text.split("/", 1)[0] if "/" in text else text)
            return out

        names = batch.get("task_name", batch.get("task", None))
        if names is None:
            return ["unknown"] * batch_size
        return [str(x) for x in names]

    def _build_rwfm_weights(
        self,
        batch: dict,
        device: torch.device,
        alpha_eff: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Build per-sample reward weights.
        Returns:
          weights: [B] tensor
          info: scalar diagnostics for logging
        """
        B = int(batch["action"].shape[0])
        uniform = torch.ones(B, dtype=torch.float32, device=device)

        if "episode_reward" not in batch or "chunk_return" not in batch:
            if not self._rwfm_missing_reward_warned:
                logger.warning(
                    "RWFM enabled but reward fields are missing in batch; using uniform weights"
                )
                self._rwfm_missing_reward_warned = True
            return uniform, {
                "rwfm_uniform_fallback": 1.0,
                "rwfm_clip_low_frac": 0.0,
                "rwfm_clip_high_frac": 0.0,
                "rwfm_clipped_frac": 0.0,
                "rwfm_weight_mean": 1.0,
                "rwfm_weight_std": 0.0,
                "rwfm_weight_min": 1.0,
                "rwfm_weight_p95": 1.0,
                "rwfm_weight_max": 1.0,
                "rwfm_effective_batch_weight": 1.0,
                "rwfm_A_mean": 0.0,
                "rwfm_A_std": 0.0,
            }

        ep = batch["episode_reward"].detach().float().cpu().numpy().reshape(-1)
        ch = batch["chunk_return"].detach().float().cpu().numpy().reshape(-1)
        groups = np.asarray(self._resolve_group_names(batch, B), dtype=object)

        z_ep = np.zeros(B, dtype=np.float32)
        z_ch = np.zeros(B, dtype=np.float32)

        for group in np.unique(groups):
            idx = np.where(groups == group)[0]
            stats = self.rwfm_group_stats.get(str(group), None)

            if stats is not None:
                med_ep = float(stats.get("episode_median", 0.0))
                sc_ep = float(stats.get("episode_scale", 1.0))
                med_ch = float(stats.get("chunk_median", 0.0))
                sc_ch = float(stats.get("chunk_scale", 1.0))
                if sc_ep < self.rwfm_eps:
                    sc_ep = 1.0
                if sc_ch < self.rwfm_eps:
                    sc_ch = 1.0
            else:
                med_ep, sc_ep = self._robust_location_scale(ep[idx])
                med_ch, sc_ch = self._robust_location_scale(ch[idx])
                if group not in self._rwfm_missing_groups_warned:
                    logger.warning(
                        "RWFM group '%s' absent in stats file; using batch-local robust stats",
                        group,
                    )
                    self._rwfm_missing_groups_warned.add(group)

            z_ep[idx] = (ep[idx] - med_ep) / sc_ep
            z_ch[idx] = (ch[idx] - med_ch) / sc_ch

        z_ep = np.clip(z_ep, -self.rwfm_z_clip, self.rwfm_z_clip)
        z_ch = np.clip(z_ch, -self.rwfm_z_clip, self.rwfm_z_clip)
        A = self.rwfm_mix_beta * z_ep + (1.0 - self.rwfm_mix_beta) * z_ch
        A = np.clip(A, -self.rwfm_a_clip, self.rwfm_a_clip)

        if alpha_eff is None:
            alpha_eff = self.rwfm_alpha

        w_raw = np.exp(float(alpha_eff) * A)
        clip_low = float(np.mean(w_raw < self.rwfm_w_min))
        clip_high = float(np.mean(w_raw > self.rwfm_w_max))

        w_clip = np.clip(w_raw, self.rwfm_w_min, self.rwfm_w_max)
        w_norm = w_clip / max(float(np.mean(w_clip)), self.rwfm_eps)
        weights = torch.from_numpy(w_norm).to(device=device, dtype=torch.float32)
        # Effective batch weight (ESS/B in [0,1]): low values mean weight collapse
        ess_ratio = float((np.sum(w_norm) ** 2) / (len(w_norm) * max(np.sum(w_norm**2), self.rwfm_eps)))

        return weights, {
            "rwfm_uniform_fallback": 0.0,
            "rwfm_clip_low_frac": clip_low,
            "rwfm_clip_high_frac": clip_high,
            "rwfm_clipped_frac": clip_low + clip_high,
            "rwfm_weight_mean": float(np.mean(w_norm)),
            "rwfm_weight_std": float(np.std(w_norm)),
            "rwfm_weight_min": float(np.min(w_norm)),
            "rwfm_weight_p95": float(np.percentile(w_norm, 95)),
            "rwfm_weight_max": float(np.max(w_norm)),
            "rwfm_effective_batch_weight": ess_ratio,
            "rwfm_A_mean": float(np.mean(A)),
            "rwfm_A_std": float(np.std(A)),
            "rwfm_alpha": float(alpha_eff),
        }

    def _build_loss_mask(self, task_texts: list[str], device: torch.device) -> torch.Tensor:
        """Build per-sample loss mask from task strings.

        Returns: [B, action_dim] float tensor, 1.0 for active joints, 0.0 for masked.
        """
        B = len(task_texts)
        if self.task_masks is None:
            return torch.ones(B, self.action_dim, device=device)

        mask = torch.zeros(B, self.action_dim, device=device)

        for i, task in enumerate(task_texts):
            matched = False
            for prefix, indices in self.task_masks.items():
                if prefix.lower() in task.lower():
                    for idx in indices:
                        mask[i, idx] = 1.0
                    matched = True
                    break
            if not matched:
                # No match — penalize all joints (safety fallback)
                mask[i, :] = 1.0
                logger.warning(f"No task mask match for: '{task}', using all joints")

        return mask

    def prepare_batch_for_smolvla(self, batch: dict) -> dict:
        """Convert our dataset batch format to SmolVLA expected format."""
        device = next(self.parameters()).device
        B = batch["state"].shape[0]

        # 1. Encode tactile and augment state
        state = batch["state"].to(device)  # [B, state_dim]

        if self.tactile_encoder.enabled:
            tactile_emb = self.tactile_encoder(
                batch["tactile_left"].to(device),
                batch["tactile_right"].to(device),
            )  # [B, 128]
            augmented_state = torch.cat([state, tactile_emb], dim=-1)
        else:
            augmented_state = state

        # Project to max_state_dim
        projected_state = self.state_augment_proj(augmented_state)  # [B, 32]

        # 2. Remap camera keys
        smolvla_batch = {}
        for our_key, smolvla_key in self.camera_remap.items():
            if our_key in batch["images"]:
                smolvla_batch[smolvla_key] = batch["images"][our_key].to(device)

        # 3. State
        smolvla_batch["observation.state"] = projected_state

        # 4. Action (pad from action_dim to max_action_dim)
        action = batch["action"].to(device)  # [B, chunk_size, action_dim]
        if action.shape[-1] < self.max_action_dim:
            pad = torch.zeros(
                B, self.chunk_size, self.max_action_dim - action.shape[-1],
                device=device, dtype=action.dtype
            )
            action_padded = torch.cat([action, pad], dim=-1)  # [B, chunk_size, 32]
        else:
            action_padded = action
        smolvla_batch["action"] = action_padded

        # 5. Tokenize task text
        task_texts = batch["task"]
        if self._tokenizer is not None:
            tokenized = self._tokenizer(
                task_texts,
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            smolvla_batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
            smolvla_batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(device)
        else:
            raise RuntimeError("SmolVLA tokenizer not available.")

        return smolvla_batch

    def forward(self, batch: dict) -> dict:
        """Forward pass for training. Supports loss masking and optional RWFM."""
        smolvla_batch = self.prepare_batch_for_smolvla(batch)

        # --- Baseline path (fully backward compatible) ---
        if not self.rwfm_enabled:
            if self.task_masks is None:
                output = self.smolvla(smolvla_batch)
                if isinstance(output, tuple):
                    loss, loss_dict = output
                    loss_dict["loss"] = loss
                    return loss_dict
                return output

            device = next(self.parameters()).device
            images, img_masks = self.smolvla.prepare_images(smolvla_batch)
            state = self.smolvla.prepare_state(smolvla_batch)
            lang_tokens = smolvla_batch["observation.language.tokens"]
            lang_masks = smolvla_batch["observation.language.attention_mask"]
            actions = self.smolvla.prepare_action(smolvla_batch)

            losses = self.smolvla.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions
            )
            losses_real = losses[:, :, :self.action_dim]

            mask = self._build_loss_mask(batch["task"], device)
            mask_expanded = mask.unsqueeze(1)
            masked_losses = losses_real * mask_expanded
            num_active = mask_expanded.sum() * losses_real.shape[1]
            loss = masked_losses.sum() / num_active.clamp(min=1)

            return {
                "loss": loss,
                "rwfm_enabled": 0.0,
                "loss_unmasked": losses_real.mean().item(),
                "loss_sample_mean": loss.item(),
                "loss_masked_unweighted": loss.item(),
                "loss_masked": loss.item(),
                "loss_rwfm": loss.item(),
                "loss_anchor": 0.0,
                "lambda_anchor": 0.0,
                "alpha_eff": 0.0,
                "rwfm_weight_mean": 1.0,
                "rwfm_weight_std": 0.0,
                "rwfm_weight_p95": 1.0,
                "rwfm_weight_max": 1.0,
                "rwfm_clipped_frac": 0.0,
                "rwfm_effective_batch_weight": 1.0,
                "mask_ratio": mask.mean().item(),
            }

        # --- RWFM path ---
        train_step = self._resolve_training_step(batch)
        alpha_eff = self._scheduled_scalar(
            base=self.rwfm_alpha,
            warmup_steps=self.rwfm_alpha_warmup_steps,
            ramp_steps=self.rwfm_alpha_ramp_steps,
            step=train_step,
        )
        lambda_anchor_eff = self._scheduled_scalar(
            base=self.anchor_lambda if self.anchor_enabled else 0.0,
            warmup_steps=self.anchor_lambda_warmup_steps,
            ramp_steps=self.anchor_lambda_ramp_steps,
            step=train_step,
        )

        device = next(self.parameters()).device
        images, img_masks = self.smolvla.prepare_images(smolvla_batch)
        state = self.smolvla.prepare_state(smolvla_batch)
        lang_tokens = smolvla_batch["observation.language.tokens"]
        lang_masks = smolvla_batch["observation.language.attention_mask"]
        actions = self.smolvla.prepare_action(smolvla_batch)

        # Raw per-element losses: [B, chunk_size, max_action_dim]
        losses = self.smolvla.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions
        )
        # Trim to true action dims
        losses_real = losses[:, :, :self.action_dim]  # [B, chunk_size, action_dim]

        # Per-sample reduction over chunk x action_dim with task mask
        mask = self._build_loss_mask(batch["task"], device)  # [B, action_dim]
        mask_expanded = mask.unsqueeze(1)  # [B, 1, action_dim]
        masked_losses = losses_real * mask_expanded
        denom_per_sample = (mask.sum(dim=1) * losses_real.shape[1]).clamp(min=1.0)
        sample_loss = masked_losses.sum(dim=(1, 2)) / denom_per_sample  # [B]

        # Reward weights and weighted reduction
        w_i, rwfm_info = self._build_rwfm_weights(batch, device, alpha_eff=alpha_eff)
        loss_rwfm = (w_i * sample_loss).sum() / w_i.sum().clamp(min=self.rwfm_eps)
        if lambda_anchor_eff > 0.0:
            loss_anchor = self._compute_anchor_loss(device)
        else:
            loss_anchor = torch.zeros((), device=device, dtype=torch.float32)
        loss_total = loss_rwfm + (lambda_anchor_eff * loss_anchor)

        loss_dict = {
            "loss": loss_total,
            "rwfm_enabled": 1.0,
            "train_step": float(train_step),
            "loss_unmasked": losses_real.mean().item(),
            "loss_sample_mean": sample_loss.mean().item(),
            "loss_masked_unweighted": sample_loss.mean().item(),
            "loss_masked": loss_total.item(),
            "loss_rwfm": loss_rwfm.item(),
            "loss_anchor": loss_anchor.item(),
            "lambda_anchor": float(lambda_anchor_eff),
            "alpha_eff": float(alpha_eff),
            "mask_ratio": mask.mean().item(),
        }
        loss_dict.update(rwfm_info)
        return loss_dict

    def predict_action(self, batch: dict) -> torch.Tensor:
        """Predict actions. Returns: [B, chunk_size, action_dim]"""
        smolvla_batch = self.prepare_batch_for_smolvla(batch)

        with torch.no_grad():
            action_pred = self.smolvla.select_action(smolvla_batch)

        # Crop back from max_action_dim to actual action_dim
        if action_pred.shape[-1] > self.action_dim:
            action_pred = action_pred[..., :self.action_dim]

        return action_pred


def build_model(cfg: dict) -> CrabSmolVLAWrapper:
    """Build the CrabSmolVLA model from config."""
    from training.tactile_encoder import DualTactileEncoder

    model_cfg = cfg["model"]

    # 1. Load SmolVLA base
    logger.info(f"Loading SmolVLA from {model_cfg['pretrained_path']}...")

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        smolvla = SmolVLAPolicy.from_pretrained(model_cfg["pretrained_path"])
        logger.info("SmolVLA loaded successfully via lerobot")
    except ImportError:
        logger.warning("lerobot not installed or SmolVLA not available.")
        smolvla = _build_placeholder_smolvla(model_cfg)

    # 2. Build tactile encoder
    tactile_encoder = DualTactileEncoder(model_cfg)
    logger.info(f"Tactile encoder: enabled={tactile_encoder.enabled}, output_dim={tactile_encoder.total_output_dim}")

    # 3. Wrap
    wrapper = CrabSmolVLAWrapper(cfg, smolvla, tactile_encoder)

    # 4. Freeze as configured
    if model_cfg.get("freeze_vision_encoder", True):
        _freeze_vision_encoder(wrapper)

    return wrapper


def _freeze_vision_encoder(wrapper: CrabSmolVLAWrapper):
    """Freeze the vision encoder weights in SmolVLA."""
    frozen_count = 0
    try:
        for name, param in wrapper.smolvla.named_parameters():
            if "vision" in name.lower() or "image_encoder" in name.lower():
                param.requires_grad = False
                frozen_count += 1
    except Exception as e:
        logger.warning(f"Could not freeze vision encoder: {e}")
    logger.info(f"Froze {frozen_count} vision encoder parameters")


class _PlaceholderSmolVLA(nn.Module):
    """Placeholder for development without lerobot installed."""
    def __init__(self, cfg):
        super().__init__()
        self.dummy = nn.Linear(32, 32)

    def forward(self, batch):
        state = batch.get("observation.state", torch.zeros(1, 32))
        loss = state.sum() * 0 + torch.tensor(1.0, requires_grad=True)
        return {"loss": loss}

    def select_action(self, batch):
        B = batch["observation.state"].shape[0]
        return torch.zeros(B, 50, 32)


def _build_placeholder_smolvla(cfg):
    return _PlaceholderSmolVLA(cfg)
