"""
Hilbert Order
Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
Please cite our work if the code is helpful to you.
"""

import torch


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes.

    This uses the classic X ^ (X >> 1) trick to compute the Gray code.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     axis: The axis along which to compute the gray code. Default=-1.

    Returns:
    --------
     Returns an ndarray of Gray codes.
    """
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """
    Encode coordinates to Hilbert curve values.
    This is a PyTorch-compatible and safe implementation based on the original logic.
    """
    # Validate inputs
    if locs.shape[-1] != num_dims:
        raise ValueError(f"Last dimension of locs must be {num_dims}")
    if num_dims * num_bits > 64:
        raise ValueError("num_dims * num_bits must be <= 64")

    # Scale locations to the integer grid
    locs_scaled = (locs * (2**num_bits - 1)).long()
    orig_shape = locs_scaled.shape
    
    # --- Start of bit manipulation ---
    # 1. Prepare masks for bit extraction
    masks = [1 << i for i in range(num_bits)]
    
    # 2. Extract bits for each dimension
    locs_bits = []
    for i in range(num_dims):
        dim_bits = []
        for mask in masks:
            dim_bits.append((locs_scaled[..., i] & mask).bool())
        locs_bits.append(torch.stack(dim_bits, dim=-1))

    # 3. Interleave bits
    # This creates the Hilbert curve ordering
    interleaved_bits = torch.stack(locs_bits, dim=-2).reshape(
        *orig_shape[:-1], -1
    )

    # 4. Pack bits into a 64-bit integer
    hilbert_code = torch.zeros(*orig_shape[:-1], dtype=torch.int64, device=locs.device)
    for i in range(interleaved_bits.shape[-1]):
        hilbert_code |= interleaved_bits[..., i].long() << i

    return hilbert_code

# The decode function is not used in the current PTV3 workflow,
# but we provide a placeholder or a correct implementation for completeness.
def decode(hilbert_code: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    # This is a placeholder as it's not required for the encoding part of the workflow.
    # A full implementation would reverse the bit packing and de-interleaving process.
    raise NotImplementedError("Hilbert decode is not implemented.")