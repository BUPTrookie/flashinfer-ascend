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
 * \file merge_chunk2_tiling.cpp
 * \brief MergeChunk2 tiling
 */

#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "merge_chunk2_tiling.h"

namespace {
constexpr uint32_t MAX_TOKEN_BYTES = 176 * 1024;
constexpr uint32_t MAX_EXPERT_BYTES = 4 * 1024;
constexpr uint32_t HIDDEN_ALIGNMENT = 16;
constexpr uint32_t TILING_KEY_BF16 = 0;
constexpr uint32_t TILING_KEY_FP16 = 1;
} // namespace

namespace optiling {
static ge::graphStatus TilingMergeChunk2(gert::TilingContext* context)
{
    MergeChunk2TilingData tiling;

    const gert::Shape& in1Chunk1Shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& in1Chunk2Shape = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape& in2Chunk1Shape = context->GetInputShape(2)->GetStorageShape();
    const gert::Shape& in2Chunk2Shape = context->GetInputShape(3)->GetStorageShape();
    const gert::Shape& groupListChunk1Shape = context->GetInputShape(4)->GetStorageShape();
    const gert::Shape& groupListChunk2Shape = context->GetInputShape(5)->GetStorageShape();

    // check dim
    if (in1Chunk1Shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (in1Chunk2Shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (in2Chunk1Shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (in2Chunk2Shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (groupListChunk1Shape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (groupListChunk2Shape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    // check shape
    const int64_t t1c1 = in1Chunk1Shape.GetDim(0);
    const int64_t h1c1 = in1Chunk1Shape.GetDim(1);
    const int64_t t1c2 = in1Chunk2Shape.GetDim(0);
    const int64_t h1c2 = in1Chunk2Shape.GetDim(1);
    const int64_t t2c1 = in2Chunk1Shape.GetDim(0);
    const int64_t h2c1 = in2Chunk1Shape.GetDim(1);
    const int64_t t2c2 = in2Chunk2Shape.GetDim(0);
    const int64_t h2c2 = in2Chunk2Shape.GetDim(1);
    const int64_t ec1 = groupListChunk1Shape.GetDim(0);
    const int64_t ec2 = groupListChunk2Shape.GetDim(0);
    if (t1c1 != t2c1) {
        return ge::GRAPH_FAILED;
    }
    if (t1c2 != t2c2) {
        return ge::GRAPH_FAILED;
    }
    if (h1c1 != h1c2) {
        return ge::GRAPH_FAILED;
    }
    if (h2c1 != h2c2) {
        return ge::GRAPH_FAILED;
    }
    if (ec1 != ec2) {
        return ge::GRAPH_FAILED;
    }

    // check range & alignment
    int maxExpert = MAX_EXPERT_BYTES / sizeof(int32_t);
    int maxHidden = MAX_TOKEN_BYTES / 2;
    if (ec1 <= 0 || ec1 > maxExpert) {
        return ge::GRAPH_FAILED;
    }
    if (h1c1 <= 0 || h2c1 <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (h1c1 % HIDDEN_ALIGNMENT != 0) {
        return ge::GRAPH_FAILED;
    }
    if (h2c1 % HIDDEN_ALIGNMENT != 0) {
        return ge::GRAPH_FAILED;
    }
    if (h1c1 > maxHidden) {
        return ge::GRAPH_FAILED;
    }
    if (h2c1 > maxHidden) {
        return ge::GRAPH_FAILED;
    }

    const int64_t maxOffset = std::numeric_limits<int32_t>::max();
    const int64_t maxRows = maxOffset / (h1c1 > h2c1 ? h1c1 : h2c1);
    if (t1c1 <= 0 || t1c2 <= 0 || t1c1 > maxRows || t1c2 > maxRows - t1c1) return ge::GRAPH_FAILED;

    auto inType = context->GetInputDesc(0)->GetDataType();
    if (inType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
    } else if (inType == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
    } else {
        return ge::GRAPH_FAILED;
    }

    tiling.set_numExpert(ec1);
    tiling.set_t1(t1c1);
    tiling.set_t2(t1c2);
    tiling.set_h1(h1c1);
    tiling.set_h2(h2c1);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (!coreNum) return ge::GRAPH_FAILED;
    tiling.set_usedCoreNum(coreNum);

    // pure vector op, will launch the same size of vector cores as SetBlockDim does.
    context->SetBlockDim(coreNum);
    auto* ws = context->GetWorkspaceSizes(1);
    if (!ws) return ge::GRAPH_FAILED;
    ws[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
