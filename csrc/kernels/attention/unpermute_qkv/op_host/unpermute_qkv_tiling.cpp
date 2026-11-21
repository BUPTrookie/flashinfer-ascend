/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unpermute_qkv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>

namespace optiling {
static ge::graphStatus TilingUnpermuteQkv(gert::TilingContext* context)
{
    const auto* input = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    const auto* nk = attrs ? attrs->GetInt(0) : nullptr;
    if (!input || !nk || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    const auto& x = input->GetStorageShape();
    if (x.GetDimNum() != 5) return ge::GRAPH_FAILED;
    const int64_t split = x.GetDim(0), s = x.GetDim(1), b = x.GetDim(2), group = x.GetDim(3), d = x.GetDim(4);
    constexpr int64_t ubElements = 180 * 1024 / 2;
    constexpr int64_t maxOffset = std::numeric_limits<int32_t>::max();
    if (split <= 0 || group <= 0 || s <= 0 || b <= 0 || d <= 0 || d % 16 || *nk <= 0 || *nk % split ||
        group > ubElements / d / split || *nk > split * group / 3 || s > maxOffset / b / d / group / split) {
        return ge::GRAPH_FAILED;
    }
    const int64_t nq = split * group - 2 * *nk;
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (nq % *nk || (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16)) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 3; ++i) {
        const auto* output = context->GetOutputShape(i);
        if (!output || context->GetOutputDesc(i)->GetDataType() != dtype) return ge::GRAPH_FAILED;
        const auto& out = output->GetStorageShape();
        if (out.GetDimNum() != 4 || out.GetDim(0) != s || out.GetDim(1) != b ||
            out.GetDim(2) != (i == 0 ? nq : *nk) || out.GetDim(3) != d) return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto cores = platform.GetCoreNumAiv();
    auto* raw = context->GetRawTilingData();
    auto* ws = context->GetWorkspaceSizes(1);
    UnpermuteQkvTilingData tiling;
    if (!cores || !ws || !raw || !raw->GetData() || raw->GetCapacity() < tiling.GetDataSize()) return ge::GRAPH_FAILED;
    tiling.set_splitSize(split);
    tiling.set_sC(s);
    tiling.set_b(b);
    tiling.set_kvGroup(nq / *nk);
    tiling.set_nK(*nk);
    tiling.set_splitNK(*nk / split);
    tiling.set_h(d);
    context->SetBlockDim(cores);
    context->SetTilingKey(dtype == ge::DT_BF16 ? 0 : 1);
    ws[0] = 16 * 1024 * 1024;
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
