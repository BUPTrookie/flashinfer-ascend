"""Group Q/K/V heads for sequence parallelism without changing their values."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec


def infer(arguments, layouts):
    q, k, v = (arguments[name] for name in ("q", "k", "v"))
    split = arguments["split_size"]
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q, k and v must have shape [S, B, heads, D]")
    s, b, nq, d = q.shape
    nkv = k.shape[2]
    if min(s, b, nq, nkv, d) <= 0:
        raise ValueError("q, k and v must have positive dimensions")
    if k.shape != v.shape or k.shape != (s, b, nkv, d):
        raise ValueError("K/V shapes must match and share S, B and D with Q")
    if nq % nkv:
        raise ValueError("the number of Q heads must be divisible by the number of KV heads")
    if type(split) is not int or split < 1 or nkv % split:
        raise ValueError("split_size must be a positive divisor of the number of KV heads")
    dtype = str(q.dtype).removeprefix("torch.")
    if dtype not in ("float16", "bfloat16") or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must share float16 or bfloat16 dtype")
    if layouts is not None and any(layout != "ND" for layout in layouts.values()):
        raise ValueError("permute_qkv requires ND tensors")
    return Invocation((q, k, v), (OutputSpec((nq + 2 * nkv, s, b, d), dtype, q.device),),
                      {"split_size": split}, InferenceSpec(compute_dtype=dtype))


OPERATOR = OperatorDefinition(
    name="permute_qkv", domain="attention", op="attention.permute_qkv",
    schema="(Tensor q, Tensor k, Tensor v, int split_size) -> Tensor",
    output_names=("output",), infer=infer,
    tensor_shapes=(("S", "B", "Nq", "D"), ("S", "B", "Nkv", "D"),
                   ("S", "B", "Nkv", "D"), ("Nout", "S", "B", "D")),
    description="Split Q/K/V heads into split_size groups, concatenate Q then K then V within each group, "
                "and return a contiguous [Nq + 2*Nkv, S, B, D] tensor.",
)
