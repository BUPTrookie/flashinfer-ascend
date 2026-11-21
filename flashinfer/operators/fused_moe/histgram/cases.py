"""Independent CPU expert counts, including expert and token tile boundaries."""


def golden(torch, ids, num_expert):
    return torch.bincount(ids.long(), minlength=num_expert).to(torch.int32)


def sample(torch, torch_npu, *, tokens=257, experts=16, pattern="random"):
    rng = torch.Generator().manual_seed(447 + tokens + experts)
    if pattern == "last":
        ids = torch.full((tokens,), experts - 1, dtype=torch.int32)
    elif pattern == "ends":
        ids = (torch.arange(tokens, dtype=torch.int32) % 2) * (experts - 1)
    else:
        ids = torch.randint(experts, (tokens,), dtype=torch.int32, generator=rng)
    return (torch_npu.npu_format_cast(ids.npu(), 2),), {"num_expert": experts}, golden(torch, ids, experts)


CASE_PARAMETERS = (
    {}, dict(tokens=1, experts=1), dict(tokens=7, experts=64, pattern="last"),
    dict(tokens=149, experts=65, pattern="ends"), dict(tokens=8203, experts=129),
    # More than 7680 tokens per AIV on both 910B and 910C. The tail must accumulate.
    dict(tokens=7680 * 64 + 17, experts=5),
)
TOLERANCES = ((0, 0),)
