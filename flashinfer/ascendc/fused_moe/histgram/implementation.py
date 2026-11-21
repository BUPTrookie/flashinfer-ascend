"""Arch22 AOT expert histogram using the framework's standard typed tensor runner."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.recipe import TensorParameter
from flashinfer.operators.author_api import DimRange, Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.operators.fused_moe.histgram.operator import OPERATOR


# All consecutive expert IDs must remain exactly representable by the kernel's fp32 compare.
MAX_EXPERT = 1 << 24
IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR, source_dir="csrc/kernels/moe/histgram", default_target="Ascend910B",
    support=Support(soc=ARCH22_SOCS, dtype="int32", layout="ND", contiguous=True,
                    cases=(CapabilityCase(name="expert_counts", compute_dtypes=("int32",)),),
                    dimensions={"T": DimRange(1, 2**31 - 1), "E": DimRange(1, MAX_EXPERT)}),
    build=BuildConfig(device="op_kernel/histgram.cpp", host=("op_host/histgram_def.cpp",),
                      op_type="FiHistgram", tiling="TilingHistgram", task_type="aiv_hard_sync",
                      sync_policy="cross_core_flags", compile_options=("--cce-auto-sync=off",)),
    binding=NativeBinding(tensors=(TensorParameter("input", "input", "int32", ("T",)),
                                   TensorParameter("counts", "output", "int32", ("E",))),
                          dtype_signatures=(("int32", "int32"),),
                          attributes=(AttributeParameter("num_expert", "int64", 1, 1, MAX_EXPERT),)),
    # 16 MiB system workspace and one 64-counter row per AIV (at most 64 on arch22).
    workspace_bound=16 * 1024 * 1024 + 64 * 64 * 4,
)
