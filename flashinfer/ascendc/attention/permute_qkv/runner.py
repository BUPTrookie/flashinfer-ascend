"""Arch22 admission ahead of the framework's standard typed ACLNN tensor runner."""

from flashinfer.ascendc._runtime import native_admission

MAX_ELEMENTS = 2**31 - 1  # signed output offsets in the imported kernel
MAX_HEADS = 4095  # maximum number of DMA blocks per copy
UB_BYTES = 180 * 1024


def admission(recipe, signature):
    reasons = native_admission(recipe, signature)
    if reasons:
        return reasons
    s, b, nq, d = signature.shapes[0]
    nkv = signature.shapes[1][2]
    try:
        split = int(dict(signature.specializations)["attribute.split_size"])
    except (KeyError, TypeError, ValueError):
        return ("permute_qkv requires an integer split_size",)
    if split < 1 or nq % nkv or nkv % split or signature.shapes[-1][0] != nq + 2 * nkv:
        return ("incompatible head counts for permute_qkv",)
    tile_rows = 2 if s * b % 2 == 0 else 1
    if tile_rows * nq * d * 2 > UB_BYTES:
        return ("the minimum Q tile exceeds the 180 KiB UB buffer",)
    if (nq + 2 * nkv) * s * b * d > MAX_ELEMENTS:
        return ("the output exceeds the signed 32-bit kernel offsets",)
    return ()
