"""
SmolVLA Configuration for Crab Robot

This configuration is optimized for:
- Input: 3 cameras (main, left_arm, right_arm) at 640x480
- Output: 14 actions (6 left arm + 6 right arm + 2 wheels)
- Hardware: Jetson Orin NX 16GB
"""

from dataclasses import dataclass, field
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def get_crab_smolvla_config() -> SmolVLAConfig:
    """
    Get SmolVLA config optimized for Crab robot.
    
    Action space (14 dims):
        - left_shoulder_pan.pos (0)
        - left_shoulder_lift.pos (1)
        - left_elbow_flex.pos (2)
        - left_wrist_flex.pos (3)
        - left_wrist_roll.pos (4)
        - left_gripper.pos (5)
        - right_shoulder_pan.pos (6)
        - right_shoulder_lift.pos (7)
        - right_elbow_flex.pos (8)
        - right_wrist_flex.pos (9)
        - right_wrist_roll.pos (10)
        - right_gripper.pos (11)
        - base_x.vel (12)
        - base_theta.vel (13)
    
    Observation space:
        - state: 14 dims (same as action for consistency)
        - images: 3 cameras at 640x480
    """
    
    # Input features: 3 cameras + state
    input_features = {
        "observation.images.main_camera": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),  # CHW format
        ),
        "observation.images.left_arm_camera": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.images.right_arm_camera": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),  # 6 left arm + 6 right arm + 2 base
        ),
    }
    
    # Output features: 14 action dims
    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(14,),
        ),
    }
    
    config = SmolVLAConfig(
        # Feature configuration
        input_features=input_features,
        output_features=output_features,
        
        # Model settings
        n_obs_steps=1,           # Single observation frame
        chunk_size=50,           # Action chunk size
        n_action_steps=50,       # Actions per inference
        
        # Dimension limits (Crab uses 14, but leave headroom)
        max_state_dim=32,
        max_action_dim=32,
        
        # Image preprocessing
        resize_imgs_with_padding=(512, 512),  # SmolVLA native resolution
        
        # Normalization
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        
        # VLM backbone - use the efficient SmolVLM2
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        num_vlm_layers=16,       # Use first 16 layers
        
        # Action expert settings
        num_expert_layers=-1,    # Match VLM layers
        expert_width_multiplier=0.75,  # Smaller expert for efficiency
        
        # Inference settings
        num_steps=10,            # Flow matching steps
        use_cache=True,          # KV cache for faster inference
        
        # Training settings (for fine-tuning)
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        
        # Optimizer settings
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-10,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=30000,
    )
    
    return config


# Convenience function to get config dict for training
def get_crab_training_config() -> dict:
    """Get training configuration dict for Crab robot."""
    return {
        "policy": {
            "type": "smolvla",
            "config": get_crab_smolvla_config(),
        },
        "robot": {
            "type": "crab",
            "cameras": ["main_camera", "left_arm_camera", "right_arm_camera"],
            "action_dim": 14,
            "state_dim": 14,
        },
        "training": {
            "batch_size": 8,      # Reduced for Orin memory
            "num_workers": 2,     # Limited workers for Orin
            "steps": 20000,
            "eval_freq": 1000,
            "save_freq": 5000,
        },
    }


if __name__ == "__main__":
    # Test configuration
    config = get_crab_smolvla_config()
    print("Crab SmolVLA Configuration:")
    print(f"  Input features: {list(config.input_features.keys())}")
    print(f"  Output features: {list(config.output_features.keys())}")
    print(f"  State dim: {config.max_state_dim}")
    print(f"  Action dim: {config.max_action_dim}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  VLM: {config.vlm_model_name}")
