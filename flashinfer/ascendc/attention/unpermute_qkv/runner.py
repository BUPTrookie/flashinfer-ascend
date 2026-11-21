"""Coupled arch22 transfer limits in addition to the generated ACLNN runner."""

from flashinfer.ascendc._runtime import native_admission

MAX_OFFSET = 2**31 - 1
UB_ELEMENTS = 180 * 1024 // 2


def admission(recipe, signature):
    reasons = native_admission(recipe, signature)
    if reasons:
        return reasons
    split, s, b, group, d = signature.shapes[0]
    nq, nk = signature.shapes[1][2], signature.shapes[2][2]
    if nq + 2 * nk != split * group or nq % nk or nk % split:
        return ("incompatible packed Q/K/V head counts",)
    if split * group * d > UB_ELEMENTS:
        return ("one packed row exceeds the 180 KiB UB",)
    if split * group * d * s * b > MAX_OFFSET:
        return ("packed input exceeds signed 32-bit offsets",)
    return ()
