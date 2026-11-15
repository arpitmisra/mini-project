"""Native MLX LoRA on query/value slices of DINOv2's fused QKV projection."""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class LoRAQKV(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.base = linear
        self.base.freeze()
        dimension = linear.weight.shape[1]
        if linear.weight.shape[0] != 3 * dimension:
            raise ValueError("Expected fused square query/key/value projections")
        self.scale = alpha / rank
        self.q_a = mx.random.normal((rank, dimension)) / math.sqrt(dimension)
        self.q_b = mx.zeros((dimension, rank))
        self.v_a = mx.random.normal((rank, dimension)) / math.sqrt(dimension)
        self.v_b = mx.zeros((dimension, rank))

    def __call__(self, x):
        q = (x @ self.q_a.T) @ self.q_b.T
        v = (x @ self.v_a.T) @ self.v_b.T
        return self.base(x) + self.scale * mx.concatenate([q, mx.zeros_like(q), v], axis=-1)

    def merged(self):
        d = self.base.weight.shape[1]
        layer = nn.Linear(d, 3 * d, bias="bias" in self.base)
        correction = mx.concatenate([self.q_b @ self.q_a, mx.zeros((d, d)), self.v_b @ self.v_a])
        layer.weight = self.base.weight + self.scale * correction
        if "bias" in self.base:
            layer.bias = self.base.bias
        return layer


def configure_backbone(backbone, settings):
    if settings.get("mode", "full") == "full":
        backbone.unfreeze()
    else:
        backbone.freeze()
        for block in backbone.blocks:
            block.attn.qkv = LoRAQKV(block.attn.qkv, settings["lora_rank"], settings["lora_alpha"])
    return backbone


def merged_parameters(backbone):
    """Export plain native weights; keep factorized training/resume state intact."""
    result = dict(tree_flatten(backbone.parameters()))
    for i, block in enumerate(backbone.blocks):
        if isinstance(block.attn.qkv, LoRAQKV):
            prefix = f"blocks.{i}.attn.qkv."
            result = {k: v for k, v in result.items() if not k.startswith(prefix)}
            result.update({prefix + k: v for k, v in tree_flatten(block.attn.qkv.merged().parameters())})
    return result
