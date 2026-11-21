"""Merge two expert-ordered chunks, with two aligned feature tensors per chunk."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

DATA = ("input1_chunk1", "input1_chunk2", "input2_chunk1", "input2_chunk2")
GROUPS = ("group_list_chunk1", "group_list_chunk2")


def infer(arguments, layouts):
    a, b, c, d = data = tuple(arguments[name] for name in DATA)
    g1, g2 = (arguments[name] for name in GROUPS)
    if any(t.ndim != 2 or min(t.shape) <= 0 for t in data):
        raise ValueError("each data chunk must have positive [tokens, hidden] dimensions")
    if a.shape[0] != c.shape[0] or b.shape[0] != d.shape[0] or a.shape[1] != b.shape[1] or c.shape[1] != d.shape[1]:
        raise ValueError("token counts must match within each chunk and hidden sizes within each feature")
    dtype = str(a.dtype).removeprefix("torch.")
    if dtype not in ("float16", "bfloat16") or any(t.dtype != a.dtype for t in data):
        raise ValueError("all four data inputs must share float16 or bfloat16 dtype")
    if g1.ndim != 1 or g1.shape != g2.shape or g1.numel() < 1:
        raise ValueError("group lists must be nonempty vectors of identical shape")
    if str(g1.dtype) != "torch.int32" or g2.dtype != g1.dtype:
        raise ValueError("group lists must have int32 dtype")
    total = a.shape[0] + b.shape[0]
    outputs = (OutputSpec((total, a.shape[1]), dtype, a.device), OutputSpec((total, c.shape[1]), dtype, a.device),
               OutputSpec(tuple(g1.shape), "int32", a.device))
    return Invocation((*data, g1, g2), outputs, {}, InferenceSpec(compute_dtype=dtype))


def validate_values(arguments):
    # These counts determine GM write addresses, so validate before launching the device kernel.
    for group, chunk in zip(GROUPS, DATA[:2]):
        counts = arguments[group].cpu().tolist()
        if min(counts) < 0 or sum(counts) != arguments[chunk].shape[0]:
            raise ValueError(f"{group} must contain nonnegative counts summing to its chunk's row count")


OPERATOR = OperatorDefinition(
    name="merge_chunk2", domain="fused_moe", op="fused_moe.merge_chunk2",
    schema="(Tensor input1_chunk1, Tensor input1_chunk2, Tensor input2_chunk1, Tensor input2_chunk2, "
           "Tensor group_list_chunk1, Tensor group_list_chunk2) -> (Tensor, Tensor, Tensor)",
    output_names=("out1", "out2", "group_list"), infer=infer, validate_values=validate_values,
    tensor_shapes=(("T1", "H1"), ("T2", "H1"), ("T1", "H2"), ("T2", "H2"), ("E",), ("E",),
                   ("T", "H1"), ("T", "H2"), ("E",)),
    description="Merge two features by expert, keeping chunk 1 before chunk 2, and return the summed expert counts.",
)
