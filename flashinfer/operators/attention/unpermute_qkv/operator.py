"""Recover contiguous Q/K/V in [S, B, heads, D] from a split-major packed input."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec


def infer(arguments, layouts):
    x, n_k = arguments["x"], arguments["n_k"]
    if x.ndim != 5 or min(x.shape) <= 0:
        raise ValueError("x requires positive [split, S, B, heads_per_split, D] dimensions")
    split, s, b, heads_per_split, d = x.shape
    if type(n_k) is not int or n_k <= 0 or n_k % split:
        raise ValueError("n_k must be positive and divisible by split")
    nq = split * heads_per_split - 2 * n_k
    if nq <= 0 or nq % n_k:
        raise ValueError("the packed heads must yield a positive multiple of n_k query heads")
    dtype = str(x.dtype).removeprefix("torch.")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("x must have float16 or bfloat16 dtype")
    outputs = tuple(OutputSpec((s, b, heads, d), dtype, x.device) for heads in (nq, n_k, n_k))
    return Invocation((x,), outputs, {"n_k": n_k}, InferenceSpec(compute_dtype=dtype))


OPERATOR = OperatorDefinition(
    name="unpermute_qkv", domain="attention", op="attention.unpermute_qkv",
    schema="(Tensor x, int n_k) -> (Tensor, Tensor, Tensor)", output_names=("q", "k", "v"), infer=infer,
    tensor_shapes=(("Split", "S", "B", "Group", "D"), ("S", "B", "Nq", "D"),
                   ("S", "B", "Nkv", "D"), ("S", "B", "Nkv", "D")),
    description="Recover Q, K and V from split-major packed heads. n_k is the total number of KV heads.",
)
