"""Arch22 AOT implementations of the two-chunk expert merge."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.recipe import TensorParameter
from flashinfer.operators.author_api import DimRange, Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.operators.fused_moe.merge_chunk2.operator import DATA, GROUPS, OPERATOR
from .runner import MAX_EXPERT, MAX_HIDDEN, MAX_OFFSET, admission


def dtype_signature(dtype):
    return (dtype,) * 4 + ("int32", "int32", dtype, dtype, "int32")


DTYPES = ("float16", "bfloat16")
SIGNATURES = tuple(dtype_signature(dtype) for dtype in DTYPES)
IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR, source_dir="csrc/kernels/moe/merge_chunk2", default_target="Ascend910B",
    support=Support(soc=ARCH22_SOCS, layout="ND", contiguous=True,
                    cases=(CapabilityCase(name="expert_merge", dtype_signatures=SIGNATURES, compute_dtypes=DTYPES),),
                    dimensions={"T1": DimRange(1, MAX_OFFSET), "T2": DimRange(1, MAX_OFFSET), "T": DimRange(2, MAX_OFFSET),
                                "H1": DimRange(16, MAX_HIDDEN, 16), "H2": DimRange(16, MAX_HIDDEN, 16),
                                "E": DimRange(1, MAX_EXPERT)}),
    build=BuildConfig(device="op_kernel/merge_chunk2.cpp", host=("op_host/merge_chunk2_def.cpp",),
                      op_type="FiMergeChunk2", tiling="TilingMergeChunk2", task_type="aiv_only",
                      sync_policy="compiler_default", workspace_bytes=0),
    binding=NativeBinding(tensors=tuple(TensorParameter(name, "output" if name in OPERATOR.output_names else "input",
                                                        dtype, shape)
                                        for name, dtype, shape in zip((*DATA, *GROUPS, *OPERATOR.output_names),
                                                                      SIGNATURES[0], OPERATOR.tensor_shapes)),
                          dtype_signatures=SIGNATURES),
    admission=admission,
)
