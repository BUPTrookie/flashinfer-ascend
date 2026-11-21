"""Arch22 AOT implementations of split-major Q/K/V unpacking."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.recipe import TensorParameter
from flashinfer.operators.author_api import DimRange, Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.operators.attention.unpermute_qkv.operator import OPERATOR
from .runner import MAX_OFFSET, UB_ELEMENTS, admission


DTYPES = ("float16", "bfloat16")
IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR, default_target="Ascend910B",
    support=Support(soc=ARCH22_SOCS, dtype=DTYPES, layout="ND", contiguous=True,
                    cases=(CapabilityCase(name="unpack_heads", compute_dtypes=DTYPES),),
                    dimensions={name: DimRange(16, UB_ELEMENTS, 16) if name == "D" else DimRange(1, maximum)
                                for name, maximum in (("Split", UB_ELEMENTS), ("S", MAX_OFFSET), ("B", MAX_OFFSET),
                                                      ("Group", UB_ELEMENTS), ("D", UB_ELEMENTS), ("Nq", UB_ELEMENTS),
                                                      ("Nkv", UB_ELEMENTS))}),
    build=BuildConfig(device="op_kernel/unpermute_qkv.cpp", host=("op_host/unpermute_qkv_def.cpp",),
                      op_type="FiUnpermuteQkv", tiling="TilingUnpermuteQkv", task_type="aiv_only",
                      sync_policy="compiler_default"),
    binding=NativeBinding(
        tensors=tuple(TensorParameter(name, "input" if name == "x" else "output", "float16", shape)
                      for name, shape in zip(("x", *OPERATOR.output_names), OPERATOR.tensor_shapes)),
        dtype_signatures=tuple((dtype,) * 4 for dtype in DTYPES),
        attributes=(AttributeParameter("n_k", "int64", 1, 1, UB_ELEMENTS),),
    ),
    admission=admission, workspace_bound=16 * 1024 * 1024,
)
