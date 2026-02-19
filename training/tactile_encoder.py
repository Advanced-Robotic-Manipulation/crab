"""
Tactile encoder module for Crab robot.
Encodes 10x10 pressure matrices into compact embeddings
that are concatenated with the robot state before feeding into SmolVLA.
"""

import torch
import torch.nn as nn


class TactileEncoder(nn.Module):
    """
    MLP encoder for 10x10 tactile sensor data.
    
    Takes flattened 100-dim tactile input and produces a compact embedding.
    Two separate encoders for left and right gripper fingers.
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        act_fn = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[activation]

        # Input normalization (tactile values range 0-4095, 12-bit ADC)
        self.input_norm = nn.LayerNorm(input_dim)

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 100] flattened tactile matrix
        Returns:
            [batch, output_dim] tactile embedding
        """
        x = self.input_norm(x)
        return self.encoder(x)


class DualTactileEncoder(nn.Module):
    """
    Dual tactile encoder for both gripper fingers.
    
    Encodes left and right tactile data independently,
    then concatenates the embeddings.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        enc_cfg = cfg.get("tactile_encoder", {})
        self.enabled = enc_cfg.get("enabled", False)

        if self.enabled:
            self.left_encoder = TactileEncoder(
                input_dim=enc_cfg.get("input_dim", 100),
                hidden_dim=enc_cfg.get("hidden_dim", 128),
                output_dim=enc_cfg.get("output_dim", 64),
                num_layers=enc_cfg.get("num_layers", 2),
                activation=enc_cfg.get("activation", "gelu"),
                dropout=enc_cfg.get("dropout", 0.1),
            )
            self.right_encoder = TactileEncoder(
                input_dim=enc_cfg.get("input_dim", 100),
                hidden_dim=enc_cfg.get("hidden_dim", 128),
                output_dim=enc_cfg.get("output_dim", 64),
                num_layers=enc_cfg.get("num_layers", 2),
                activation=enc_cfg.get("activation", "gelu"),
                dropout=enc_cfg.get("dropout", 0.1),
            )
            self.total_output_dim = enc_cfg.get("output_dim", 64) * 2
        else:
            self.total_output_dim = 0

    def forward(
        self, tactile_left: torch.Tensor, tactile_right: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Args:
            tactile_left: [batch, 100]
            tactile_right: [batch, 100]
        Returns:
            [batch, output_dim * 2] concatenated tactile embeddings, or None if disabled
        """
        if not self.enabled:
            return None

        left_emb = self.left_encoder(tactile_left)
        right_emb = self.right_encoder(tactile_right)
        return torch.cat([left_emb, right_emb], dim=-1)
