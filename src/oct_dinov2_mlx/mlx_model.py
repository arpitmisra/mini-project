"""DINOv2 with differentiable positional interpolation on CPU and Apple GPU."""
import math

import mlx.core as mx
from mlx_vlm.models.dinov2.dinov2 import Model
from mlx_vlm.models.interpolate import _bicubic_weights_1d


def resize_positions(x, height, width):
    """NHWC bicubic resize: half-pixel sampling, a=-0.75, no antialias.

    The pinned adapter's tap tables match the HF checkpoint. Use only ordinary
    MLX gathers/reductions here: its automatic Metal custom kernel has no VJP,
    so selecting it on a Mac breaks gradients to the trainable position table.
    """
    _, in_height, in_width, _ = x.shape
    if (in_height, in_width) == (height, width):
        return x
    iy, wy = _bicubic_weights_1d(in_height, height, height / in_height)
    ix, wx = _bicubic_weights_1d(in_width, width, width / in_width)
    # Width then height; repeated border indices accumulate gradients naturally.
    x = mx.sum(mx.take(x, ix, axis=2) * wx[None, None, :, :, None], axis=3)
    return mx.sum(mx.take(x, iy, axis=1) * wy[None, :, :, None, None], axis=2)


class DifferentiableDINOv2(Model):
    """Preserves upstream parameter names and strict checkpoint loading."""

    def interpolate_pos_encoding(self, x, h, w):
        if self.config.interpolate_offset or self.config.interpolate_antialias:
            raise ValueError("Only the pinned HF size-based, non-antialiased positions are supported")
        count = self.pos_embed.shape[1] - 1
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("Expected a square pretrained position grid")
        height, width = h // self.patch_size, w // self.patch_size
        if x.shape[1] - 1 != height * width:
            raise ValueError("Image dimensions and patch-token count disagree")
        positions = self.pos_embed[:, 1:].reshape(1, side, side, x.shape[-1])
        positions = resize_positions(positions.astype(mx.float32), height, width)
        return mx.concatenate(
            [self.pos_embed[:, :1], positions.reshape(1, height * width, x.shape[-1])], axis=1
        ).astype(x.dtype)
