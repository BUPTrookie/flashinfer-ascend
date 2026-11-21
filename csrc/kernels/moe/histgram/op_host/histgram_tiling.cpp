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
 * \file histgram_tiling.cpp
 * \brief Histgram tiling
 */
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "histgram_tiling.h"

namespace optiling {
namespace {
// index
constexpr int NUM_EXPERT_INDEX = 0;

// some constants
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t EXPERT_CHUNK_SIZE = 64;
} // namespace

static ge::graphStatus TilingHistgram(gert::TilingContext* context)
{
    HistgramTilingData tiling;

    const gert::Shape& input_shape = context->GetInputShape(0)->GetStorageShape();
    // check dim
    if (input_shape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const auto numToken = input_shape.GetDim(0);
    if (numToken < 0 || numToken > std::numeric_limits<int32_t>::max()) return ge::GRAPH_FAILED;
    tiling.set_numToken(numToken);

    const auto* attrs = context->GetAttrs();
    const auto* numExpert = attrs ? attrs->GetInt(NUM_EXPERT_INDEX) : nullptr;
    // Expert IDs are compared in fp32 in the vector kernel.
    if (!numExpert || *numExpert < 1 || *numExpert > (1 << 24)) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_numExpert(*numExpert);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (!coreNum) return ge::GRAPH_FAILED;
    tiling.set_usedCoreNum(coreNum);

    // pure vector op, will launch the same size of vector cores as SetBlockDim does.
    context->SetBlockDim(coreNum);
    context->SetTilingKey(0);
    context->SetScheduleMode(1); // 1: batchmode模式

    // set workspace
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_SIZE + EXPERT_CHUNK_SIZE * sizeof(int32_t) * coreNum;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
