"""
CrabSmolVLA: SmolVLA wrapper adapted for Crab robot.

v3: Loss masking support for unified 14-DOF training.
    Active joints per task defined in config task_masks.
    Configurable action_dim/state_dim (supports joint subsets).
    Tactile encoder optional (disabled via config).
"""

import logging
from pathlib import Path

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

    def _build_loss_mask(self, task_texts: list[str], device: torch.device) -> torch.Tensor:
        """Build per-sample loss mask from task strings.

        Returns: [B, action_dim] float tensor, 1.0 for active joints, 0.0 for masked.
        """
        B = len(task_texts)
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
        """Forward pass for training. Supports loss masking when task_masks is configured."""
        smolvla_batch = self.prepare_batch_for_smolvla(batch)

        if self.task_masks is None:
            # Original behavior — no masking
            output = self.smolvla(smolvla_batch)
            if isinstance(output, tuple):
                loss, loss_dict = output
                loss_dict["loss"] = loss
                return loss_dict
            return output

        # === Loss-masked forward ===
        # Get per-element losses from SmolVLA's flow matching
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

        # Trim to real action dims: [B, chunk_size, action_dim]
        losses_real = losses[:, :, :self.action_dim]

        # Build mask: [B, action_dim] -> [B, 1, action_dim]
        mask = self._build_loss_mask(batch["task"], device)
        mask_expanded = mask.unsqueeze(1)  # [B, 1, action_dim]

        # Apply mask and reduce
        masked_losses = losses_real * mask_expanded
        # Mean over active elements: sum / (num_active_elements * chunk_size)
        num_active = mask_expanded.sum() * losses_real.shape[1]  # total active cells
        loss = masked_losses.sum() / num_active.clamp(min=1)

        loss_dict = {
            "loss": loss,
            "loss_unmasked": losses_real.mean().item(),
            "loss_masked": loss.item(),
            "mask_ratio": mask.mean().item(),
        }
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
