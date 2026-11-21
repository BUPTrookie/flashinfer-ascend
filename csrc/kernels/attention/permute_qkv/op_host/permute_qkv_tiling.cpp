/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file permute_qkv_tiling.cpp
 * \brief
 */

#include "permute_qkv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>

namespace optiling {
static ge::graphStatus TilingPermuteQkv(gert::TilingContext* context)
{
    const auto* q = context->GetInputShape(0);
    const auto* k = context->GetInputShape(1);
    const auto* v = context->GetInputShape(2);
    const auto* out = context->GetOutputShape(0);
    const auto* attrs = context->GetAttrs();
    const auto* split = attrs ? attrs->GetInt(0) : nullptr;
    if (!q || !k || !v || !out || !split || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto& qs = q->GetStorageShape();
    const auto& ks = k->GetStorageShape();
    const auto& vs = v->GetStorageShape();
    const auto& os = out->GetStorageShape();
    if (qs.GetDimNum() != 4 || ks.GetDimNum() != 4 || vs.GetDimNum() != 4 || os.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < 4; ++axis) {
        if (qs.GetDim(axis) <= 0 || ks.GetDim(axis) != vs.GetDim(axis) ||
            (axis != 2 && qs.GetDim(axis) != ks.GetDim(axis))) {
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t s = qs.GetDim(0), b = qs.GetDim(1), nq = qs.GetDim(2), d = qs.GetDim(3), nkv = ks.GetDim(2);
    constexpr int64_t maxOffset = std::numeric_limits<int32_t>::max();
    constexpr int64_t ubBytes = 180 * 1024;
    if (nq > 4095 || nkv <= 0 || nq % nkv || *split <= 0 || nkv % *split || d % 16 ||
        s > maxOffset / b / d / (nq + 2 * nkv)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tileRows = s * b % 2 == 0 ? 2 : 1;
    if (d > ubBytes / 2 / tileRows / nq || os.GetDim(0) != nq + 2 * nkv ||
        os.GetDim(1) != s || os.GetDim(2) != b || os.GetDim(3) != d) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDesc(0)->GetDataType();
    if ((dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16) || context->GetInputDesc(1)->GetDataType() != dtype ||
        context->GetInputDesc(2)->GetDataType() != dtype || context->GetOutputDesc(0)->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    PermuteQkvTilingData tiling;
    if (!cores || !workspace || !raw || !raw->GetData() || raw->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_splitSize(*split);
    tiling.set_sC(s);
    tiling.set_b(b);
    tiling.set_kvGroup(nq / nkv);
    tiling.set_nK(nkv);
    tiling.set_splitNK(nkv / *split);
    tiling.set_h(d);
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(cores);
    context->SetTilingKey(dtype == ge::DT_BF16 ? 0 : 1);
    workspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
