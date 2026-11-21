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
 * \file permute_qkv_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include <limits>

namespace ops {
static ge::graphStatus InferShapePermuteQkv(gert::InferShapeContext* context)
{
    const auto* q = context->GetInputShape(0);
    const auto* k = context->GetInputShape(1);
    const auto* v = context->GetInputShape(2);
    auto* out = context->GetOutputShape(0);
    if (!q || !k || !v || !out || q->GetDimNum() != 4 || k->GetDimNum() != 4 || v->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const int64_t nq = q->GetDim(2);
    const int64_t nkv = k->GetDim(2);
    if (nq >= 0 && nkv > (std::numeric_limits<int64_t>::max() - nq) / 2) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(4);
    out->SetDim(0, nq < 0 || nkv < 0 ? -1 : nq + 2 * nkv);
    out->SetDim(1, q->GetDim(0));
    out->SetDim(2, q->GetDim(1));
    out->SetDim(3, q->GetDim(3));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePermuteQkv(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}
} // namespace ops
