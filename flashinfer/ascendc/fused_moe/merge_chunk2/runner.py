"""Check the signed element offsets used by arch22 before selecting its standard runner."""

from flashinfer.ascendc._runtime import native_admission

MAX_OFFSET = 2**31 - 1
MAX_HIDDEN = 176 * 1024 // 2
MAX_EXPERT = 1024


def admission(recipe, signature):
    reasons = native_admission(recipe, signature)
    if reasons:
        return reasons
    t1, h1 = signature.shapes[0]
    t2, h2 = signature.shapes[3]
    if signature.shapes[6][0] != t1 + t2 or (t1 + t2) * max(h1, h2) > MAX_OFFSET:
        return ("merged token count or signed 32-bit output offset is invalid",)
    return ()
