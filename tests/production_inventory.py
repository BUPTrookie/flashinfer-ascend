"""Independently reviewed release expectations. Never generated from the catalog.

New production entries require an explicit review here. Every consumer retains
exact count/set equality; authoring/build tools must never update this file.
"""

PRODUCTION = (
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend910C", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend910C"),
    ("ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910C", "gemm.grouped_matmul_swiglu_quant", "grouped_matmul_swiglu_quant", "native", "Ascend910C"),
    ("ascendc/attention/permute_qkv/native/Ascend910C", "attention.permute_qkv", "permute_qkv", "native", "Ascend910C"),
    ("ascendc/attention/unpermute_qkv/native/Ascend910C", "attention.unpermute_qkv", "unpermute_qkv", "native", "Ascend910C"),
    ("ascendc/fused_moe/histgram/native/Ascend910C", "fused_moe.histgram", "histgram", "native", "Ascend910C"),
    ("ascendc/fused_moe/merge_chunk2/native/Ascend910C", "fused_moe.merge_chunk2", "merge_chunk2", "native", "Ascend910C"),
    ("ascendc/attention/sparse_flash_attention/native/Ascend910C", "attention.sparse_flash_attention", "sparse_flash_attention", "native", "Ascend910C"),
    ("ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "gemm.matmul_bias_silu", "matmul_bias_silu", "native", "Ascend950"),
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend910B", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend910B"),
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend950", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend950"),
    ("ascendc/gemm/matmul/native/op_package/Ascend950", "gemm.matmul", "matmul", "native", "Ascend950"),
    ("ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B", "gemm.grouped_matmul_swiglu_quant", "grouped_matmul_swiglu_quant", "native", "Ascend910B"),
    ("ascendc/attention/permute_qkv/native/Ascend910B", "attention.permute_qkv", "permute_qkv", "native", "Ascend910B"),
    ("ascendc/attention/unpermute_qkv/native/Ascend910B", "attention.unpermute_qkv", "unpermute_qkv", "native", "Ascend910B"),
    ("ascendc/fused_moe/histgram/native/Ascend910B", "fused_moe.histgram", "histgram", "native", "Ascend910B"),
    ("ascendc/fused_moe/merge_chunk2/native/Ascend910B", "fused_moe.merge_chunk2", "merge_chunk2", "native", "Ascend910B"),
    ("ascendc/attention/sparse_flash_attention/native/Ascend910B", "attention.sparse_flash_attention", "sparse_flash_attention", "native", "Ascend910B"),
)
PRODUCTION_KERNEL_IDS = frozenset(row[0] for row in PRODUCTION)
PRODUCTION_OPS = frozenset(row[1] for row in PRODUCTION)
DEFAULT_DESCRIPTOR_COUNT = 98
LEGACY_DESCRIPTOR_COUNT = 81
STACK_COUNTS = {"ascendc": 17, "cann": 41, "reference": 38, "triton": 2}
PUBLIC_RECIPES = frozenset((r[2], r[3], r[4], 2, "op_package") for r in PRODUCTION) | frozenset({
    ("matmul_probe", "native", "Ascend950", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend950", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend910B", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend910C", 2, "op_package"),
    ("matmul_probe", "native", "Ascend910B", 2, "op_package"),
    ("matmul_probe", "native", "Ascend910C", 2, "op_package"),
    ("matmul_probe", "native", "ascend910b1", 2, "op_package"),
    ("matmul_probe", "native", "ascend910_9391", 2, "op_package"),
    ("matmul_probe", "native", "ascend310p1", 2, "op_package"),
})
