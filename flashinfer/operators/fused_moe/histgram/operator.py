"""Count token assignments for each expert."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec


def infer(arguments, layouts):
    ids, experts = arguments["input"], arguments["num_expert"]
    if ids.ndim != 1 or ids.shape[0] == 0:
        raise ValueError("input must be a nonempty one-dimensional expert ID tensor")
    if str(ids.dtype) != "torch.int32":
        raise ValueError("expert IDs must have int32 dtype")
    if type(experts) is not int or experts <= 0:
        raise ValueError("num_expert must be a positive integer")
    return Invocation((ids,), (OutputSpec((experts,), "int32", ids.device),), {"num_expert": experts},
                      InferenceSpec(compute_dtype="int32"))


def validate_values(arguments):
    ids = arguments["input"]
    if ((ids < 0) | (ids >= arguments["num_expert"])).any().item():
        raise ValueError("expert IDs must be in [0, num_expert)")


OPERATOR = OperatorDefinition(
    name="histgram", domain="fused_moe", op="fused_moe.histgram",
    schema="(Tensor input, int num_expert) -> Tensor", output_names=("counts",), infer=infer,
    validate_values=validate_values, tensor_shapes=(("T",), ("E",)),
    description="Count the occurrences of each expert ID in a nonempty int32 token vector.",
)
