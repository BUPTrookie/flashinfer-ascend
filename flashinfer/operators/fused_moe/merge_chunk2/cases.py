"""CPU reference uses expert-wise splits and concatenation, independent of native offsets."""


def golden(torch, a, b, c, d, g1, g2):
    def merge(x, y):
        chunks = zip(x.split(g1.tolist()), y.split(g2.tolist()))
        return torch.cat([part for pair in chunks for part in pair], dim=0)
    return merge(a, b), merge(c, d), g1 + g2


def sample(torch, torch_npu, *, counts1=(2, 0, 3, 1), counts2=(1, 4, 0, 1), h1=32, h2=64, dtype="float16"):
    g1, g2 = (torch.tensor(counts, dtype=torch.int32) for counts in (counts1, counts2))
    rng = torch.Generator().manual_seed(552 + sum(counts1) + h1)
    data = tuple(torch.randn((sum(counts), hidden), generator=rng).to(getattr(torch, dtype))
                 for hidden in (h1, h2) for counts in (counts1, counts2))
    inputs = (*data, g1, g2)
    return tuple(torch_npu.npu_format_cast(t.npu(), 2) for t in inputs), {}, golden(torch, *inputs)


CASE_PARAMETERS = (
    {}, dict(counts1=(1,), counts2=(1,), h1=16, h2=16),
    dict(counts1=(0, 1, 7, 0, 2), counts2=(0, 3, 0, 0, 4), dtype="bfloat16"),
    dict(counts1=(130, 0, 1), counts2=(2, 149, 0), h1=128, h2=16),
    # One full expert buffer, with populated experts at opposite ends.
    dict(counts1=(1,) + (0,) * 1023, counts2=(0,) * 1023 + (2,), h1=16, h2=32, dtype="bfloat16"),
    # More than one UB transfer per active core, with different hidden widths.
    dict(counts1=(59, 1, 53), counts2=(0, 102, 1), h1=32768, h2=128, dtype="bfloat16"),
)
TOLERANCES = ((0, 0),) * 3
