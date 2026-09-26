# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for restoring MoE activation scale layouts after a2a dispatch."""

import torch

from vllm.utils.flashinfer import (
    nvfp4_block_scale_interleave as block_scale_interleave,
)

# Quant dtypes whose activation scales use the swizzled 128x4 layout when the
# kernel declares is_scale_swizzled=True. Must cover every quant_dtype branch in
# moe_kernel_quantize_input that honors is_scale_swizzled.
_SWIZZLED_SCALE_DTYPES = ("nvfp4", "mxfp8")


def restore_dispatched_scale_layout(
    a1q_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    is_scale_swizzled: bool,
) -> torch.Tensor | None:
    """Swizzle activation scales after an a2a dispatch, if the kernel
    expects it.

    Dispatch paths quantize with is_scale_swizzled=False, since the swizzled
    layout is padded and would not line up row-for-row with the hidden states
    in the a2a. For kernels that expect swizzled scales, convert the
    dispatched row-major scales to the swizzled 128x4 layout here; otherwise
    return the scales unchanged.
    """
    if (
        a1q_scale is None
        or not is_scale_swizzled
        or quant_dtype not in _SWIZZLED_SCALE_DTYPES
    ):
        return a1q_scale
    if a1q_scale.element_size() == 1:
        a1q_scale = a1q_scale.view(torch.uint8)
    # despite its name, this is flashinfer's generic block_scale_interleave
    # and handles mxfp8 scale vectors as well
    return block_scale_interleave(a1q_scale)
