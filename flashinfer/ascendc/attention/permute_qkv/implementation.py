"""Arch22 vector-core AOT implementations of the logical Q/K/V permutation."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import TensorParameter
from flashinfer.operators.attention.permute_qkv.operator import OPERATOR
from flashinfer.operators.author_api import DimRange, Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import MAX_ELEMENTS, MAX_HEADS, UB_BYTES, admission


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR, default_target="Ascend910B",
    support=Support(soc=ARCH22_SOCS, dtype=("float16", "bfloat16"), layout="ND", contiguous=True,
                    cases=(CapabilityCase(name="head_permutation", compute_dtypes=("float16", "bfloat16")),),
                    dimensions={"S": DimRange(1, MAX_ELEMENTS), "B": DimRange(1, MAX_ELEMENTS),
                                "Nq": DimRange(1, MAX_HEADS), "Nkv": DimRange(1, MAX_HEADS),
                                "Nout": DimRange(3, 3 * MAX_HEADS), "D": DimRange(16, UB_BYTES // 2, 16)}),
    build=BuildConfig(device="op_kernel/permute_qkv.cpp", host=("op_host/permute_qkv_def.cpp",),
                      op_type="FiPermuteQkv", tiling="TilingPermuteQkv", task_type="aiv_only",
                      sync_policy="compiler_default"),
    binding=NativeBinding(
        tensors=tuple(TensorParameter(name, role, "float16", shape) for (name, role), shape in zip(
            (("q", "input"), ("k", "input"), ("v", "input"), ("output", "output")), OPERATOR.tensor_shapes)),
        dtype_signatures=(("float16",) * 4, ("bfloat16",) * 4),
        attributes=(AttributeParameter("split_size", "int64", 1, 1, MAX_HEADS),),
    ),
    admission=admission, workspace_bound=16 * 1024 * 1024,
)
