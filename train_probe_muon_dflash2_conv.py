"""Muon probe for the convolution-only DFlash2 architecture ablation.

The convolution follows NVIDIA NeMo AutoModel's DFlash2 implementation.  This
ablation intentionally excludes the candidate selector so its measurements can
be compared directly with the existing DFlash run.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from train_probe import main


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    """Apply content-adaptive depthwise taps without crossing block boundaries."""
    if hidden.ndim != 3:
        raise ValueError("DFlash2 convolution expects [batch, sequence, hidden]")
    batch, sequence_length, hidden_size = hidden.shape
    if block_size < 1 or sequence_length % block_size != 0:
        raise ValueError(
            f"block_size={block_size} must divide sequence_length={sequence_length}"
        )
    if hidden_size % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must divide hidden_size={hidden_size}"
        )

    groups = hidden_size // group_size
    kernel_size = base_kernel.shape[0]
    blocks = hidden.view(
        batch,
        sequence_length // block_size,
        block_size,
        groups,
        group_size,
    )
    dynamic = dynamic.view(
        batch,
        sequence_length // block_size,
        block_size,
        kernel_size,
        groups,
        1,
    )
    output = torch.zeros_like(blocks)
    for offset in range(kernel_size):
        values = (
            blocks
            if offset == 0
            else F.pad(blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0))
        )
        kernel = base_kernel[offset].view(1, 1, 1, groups, group_size)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, :, offset], values)
    return output.view(batch, sequence_length, hidden_size)


class GroupedDynamicCausalConv(nn.Module):
    """DFlash2's shared two-tap dynamic convolution around one sublayer."""

    def __init__(
        self, hidden_size: int, kernel_size: int = 2, group_size: int = 16
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if hidden_size % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide hidden_size={hidden_size}"
            )
        groups = hidden_size // group_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * kernel_size * groups,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.kernel_projection.weight)
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0, :] = 1

    def prepare(
        self, hidden: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1],
            2,
            self.kernel_size,
            groups,
        )
        convolved = _grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
            block_size,
        )
        return convolved, dynamic[..., 1, :, :]

    def finish(
        self,
        hidden: torch.Tensor,
        dynamic: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        return _grouped_dynamic_convolve(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.group_size,
            block_size,
        )


class DFlash2ConvDecoderLayer(nn.Module):
    """Add DFlash2 convolutions to an already constructed DFlash layer."""

    def __init__(self, layer: nn.Module, block_size: int) -> None:
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.block_size = block_size
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.attention_conv = GroupedDynamicCausalConv(self.hidden_size)
        self.mlp_conv = GroupedDynamicCausalConv(self.hidden_size)

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_dynamic = self.attention_conv.prepare(
            hidden_states, self.block_size
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states, attention_dynamic, self.block_size
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_dynamic = self.mlp_conv.prepare(
            hidden_states, self.block_size
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(
            hidden_states, mlp_dynamic, self.block_size
        )
        return residual + hidden_states


def add_dflash2_convolution(draft: nn.Module) -> nn.Module:
    """Replace each draft layer with its identity-initialized conv variant."""
    if not hasattr(draft, "layers") or not draft.layers:
        raise ValueError("draft has no decoder layers")
    reference = next(draft.parameters())
    wrapped_layers = []
    for layer in draft.layers:
        wrapped = DFlash2ConvDecoderLayer(layer, draft.block_size)
        wrapped.attention_conv.to(device=reference.device, dtype=reference.dtype)
        wrapped.mlp_conv.to(device=reference.device, dtype=reference.dtype)
        wrapped_layers.append(wrapped)
    draft.layers = nn.ModuleList(wrapped_layers)
    return draft


if __name__ == "__main__":
    main(
        optimizer_name="muon",
        loss_name="pal",
        pal_ce_weight=0.10,
        architecture_name="dflash2_conv",
        draft_transform=add_dflash2_convolution,
    )
