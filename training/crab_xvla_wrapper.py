"""
CrabXVLA: X-VLA wrapper adapted for Crab robot with tactile sensing.

This wraps the pretrained X-VLA model and adds:
1. Tactile encoder (DualTactileEncoder) for 10x10 pressure matrices
2. Proprio augmentation: robot state (14) + tactile embeddings (128) -> projected to max_action_dim (20)
3. Overrides action_space to "auto" mode with real_dim=14, max_dim=20
4. Compatible with X-VLA's flow-matching training and inference

X-VLA forward expects:
    input_ids, image_input, image_mask, domain_id, proprio, action

X-VLA generate_actions expects:
    input_ids, image_input, image_mask, domain_id, proprio, steps
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrabXVLAWrapper(nn.Module):
    """
    Wraps X-VLA model with tactile encoding for the Crab bimanual robot.

    Architecture:
        Input:
            - 3 camera images (main, left_arm, right_arm) -> Florence2 vision encoder
            - Robot state [14] + tactile embeddings [128] -> proprio projection -> [20]
            - Task text -> BartTokenizer -> language tokens
        
        X-VLA internals:
            - Florence2 encodes images + language -> contextual features
            - SoftPromptedTransformer w/ domain-aware soft prompts
            - Flow-matching action decoder generates [num_actions, 20] actions
        
        Output:
            - Action chunk [num_actions, real_action_dim=14] (trimmed from 20)
    """

    def __init__(self, cfg: dict, xvla_model, xvla_processor, tactile_encoder):
        super().__init__()
        self.cfg = cfg
        self.xvla = xvla_model
        self.processor = xvla_processor
        self.tactile_encoder = tactile_encoder

        model_cfg = cfg["model"]
        self.real_action_dim = model_cfg.get("real_action_dim", 14)
        self.max_action_dim = model_cfg.get("max_action_dim", 20)
        self.num_actions = model_cfg.get("num_actions", 30)
        self.domain_id_value = model_cfg.get("domain_id", 0)

        # Proprio augmentation projection:
        # robot_state (14) + tactile_left_emb (64) + tactile_right_emb (64) = 142
        # -> project to max_action_dim (20) to match X-VLA's expected proprio dim
        tactile_dim = self.tactile_encoder.total_output_dim if self.tactile_encoder.enabled else 0
        augmented_proprio_dim = self.real_action_dim + tactile_dim

        self.proprio_proj = nn.Linear(augmented_proprio_dim, self.max_action_dim)
        nn.init.xavier_uniform_(self.proprio_proj.weight)
        nn.init.zeros_(self.proprio_proj.bias)

        logger.info(
            f"CrabXVLAWrapper: real_action_dim={self.real_action_dim}, "
            f"tactile_dim={tactile_dim}, augmented_proprio={augmented_proprio_dim} -> {self.max_action_dim}"
        )

    def augment_proprio(self, proprio: torch.Tensor, tactile_left: torch.Tensor, tactile_right: torch.Tensor) -> torch.Tensor:
        """
        Augment proprioception with tactile embeddings and project.
        
        Args:
            proprio: [B, max_action_dim] (14 real + 6 padding)
            tactile_left: [B, 100]
            tactile_right: [B, 100]
        Returns:
            [B, max_action_dim] augmented proprio
        """
        device = proprio.device
        # Extract real proprio (first 14 dims)
        real_proprio = proprio[..., :self.real_action_dim]  # [B, 14]

        if self.tactile_encoder.enabled:
            tactile_emb = self.tactile_encoder(
                tactile_left.to(device),
                tactile_right.to(device),
            )  # [B, 128]
            augmented = torch.cat([real_proprio, tactile_emb], dim=-1)  # [B, 142]
        else:
            augmented = real_proprio  # [B, 14]

        # Project to max_action_dim
        projected = self.proprio_proj(augmented)  # [B, 20]
        return projected

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        tactile_left: torch.Tensor,
        tactile_right: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with tactile augmentation.
        
        Args:
            input_ids: [B, L] language tokens
            image_input: [B, V, C, H, W] multi-view images
            image_mask: [B, V] valid image mask
            domain_id: [B] domain indices
            proprio: [B, max_action_dim] proprioception (padded)
            action: [B, T, max_action_dim] ground truth actions (padded)
            tactile_left: [B, 100] left tactile data
            tactile_right: [B, 100] right tactile data
        
        Returns:
            Dict of loss tensors from X-VLA's action_space.compute_loss
        """
        # Augment proprio with tactile
        augmented_proprio = self.augment_proprio(proprio, tactile_left, tactile_right)

        # Forward through X-VLA
        return self.xvla(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=augmented_proprio,
            action=action,
        )

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        tactile_left: torch.Tensor,
        tactile_right: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate actions using iterative denoising.
        
        Returns:
            [B, num_actions, real_action_dim] predicted actions (trimmed to 14)
        """
        augmented_proprio = self.augment_proprio(proprio, tactile_left, tactile_right)

        actions = self.xvla.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=augmented_proprio,
            steps=steps,
        )  # [B, num_actions, max_action_dim] already postprocessed

        # Trim to real action dim
        return actions[..., :self.real_action_dim]

    def predict_action(self, batch: dict, steps: int = 10) -> torch.Tensor:
        """
        High-level predict from a raw batch dict (for visualization/eval).
        Returns: [B, num_actions, real_action_dim]
        """
        device = next(self.parameters()).device

        # Encode language
        lang = self.processor.encode_language(batch["language_instruction"])
        input_ids = lang["input_ids"].to(device)

        return self.generate_actions(
            input_ids=input_ids,
            image_input=batch["image_input"].to(device),
            image_mask=batch["image_mask"].to(device),
            domain_id=batch["domain_id"].to(device),
            proprio=batch["proprio"].to(device),
            tactile_left=batch["tactile_left"].to(device),
            tactile_right=batch["tactile_right"].to(device),
            steps=steps,
        )


def _patch_action_space(xvla, real_action_dim: int, max_action_dim: int):
    """
    Patch X-VLA's action_space from pretrained (ee6d, dim=20) to auto mode
    that properly handles our 14 DoF robot.
    
    This is critical: without this patch, the model would:
    - compute_loss on all 20 dims (including 6 meaningless padding dims)
    - postprocess with sigmoid on gripper indices 9,19 (wrong for our robot)
    
    With the patch:
    - compute_loss ONLY on first 14 dims (real robot joints)
    - postprocess trims output from 20 to 14
    - preprocess pads input from 14 to 20
    - No sigmoid/BCE — pure MSE on all joints (joint-space control)
    """
    from models.action_hub import build_action_space

    logger.info(
        f"Patching action_space: {xvla.action_mode} (dim={xvla.action_space.dim_action}) "
        f"-> auto (real_dim={real_action_dim}, max_dim={max_action_dim})"
    )

    # Build new auto action space
    new_action_space = build_action_space(
        "auto",
        real_dim=real_action_dim,
        max_dim=max_action_dim,
    )

    # Replace action space in model
    xvla.action_space = new_action_space
    xvla.action_mode = "auto"

    # Update config to reflect change
    xvla.config.action_mode = "auto"
    xvla.config.real_action_dim = real_action_dim
    xvla.config.max_action_dim = max_action_dim

    logger.info(
        f"Action space patched: mode=auto, dim_action={new_action_space.dim_action}, "
        f"real_dim={new_action_space.real_dim}"
    )
    logger.info(
        f"  Loss: MSE on first {real_action_dim} dims only"
    )
    logger.info(
        f"  Inference: model outputs {max_action_dim} dims -> trimmed to {real_action_dim}"
    )

    return xvla


def build_model(cfg: dict) -> CrabXVLAWrapper:
    """Build the CrabXVLA model from config."""
    from training.tactile_encoder import DualTactileEncoder

    model_cfg = cfg["model"]
    pretrained_path = model_cfg["pretrained_path"]
    real_action_dim = model_cfg.get("real_action_dim", 14)
    max_action_dim = model_cfg.get("max_action_dim", 20)

    # 1. Load X-VLA model
    logger.info(f"Loading X-VLA from {pretrained_path}...")

    try:
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        xvla = XVLA.from_pretrained(pretrained_path)
        processor = XVLAProcessor.from_pretrained(pretrained_path)
        logger.info(f"X-VLA loaded successfully: {pretrained_path}")
        logger.info(f"  Original action_mode={xvla.action_mode}, num_actions={xvla.num_actions}")
        logger.info(f"  Original action_space.dim_action={xvla.action_space.dim_action}")

    except ImportError as e:
        logger.error(
            f"Failed to import X-VLA modules: {e}. "
            f"Make sure the X-VLA repo (https://github.com/2toinf/X-VLA) is cloned "
            f"and its root is in your Python path."
        )
        raise

    # 2. CRITICAL: Patch action space from pretrained (ee6d) to auto (14 real DoF)
    #    Pretrained model uses ee6d (20 dims: xyz+rot6d+gripper for 2 arms)
    #    Our Crab robot uses joint-space (14 dims: 6+6 arms + 2 base)
    #    "auto" mode:
    #      - Loss on first 14 dims only (ignores padding)
    #      - No sigmoid on grippers (MSE for all joints)
    #      - Trims output from 20 to 14 at inference
    xvla = _patch_action_space(xvla, real_action_dim, max_action_dim)

    # 3. Build tactile encoder
    tactile_encoder = DualTactileEncoder(model_cfg)
    logger.info(f"Tactile encoder: enabled={tactile_encoder.enabled}, output_dim={tactile_encoder.total_output_dim}")

    # 4. Wrap
    wrapper = CrabXVLAWrapper(cfg, xvla, processor, tactile_encoder)

    # 5. Count parameters
    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    return wrapper
