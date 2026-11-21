/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <limits>

namespace ops {
static ge::graphStatus InferShapeUnpermuteQkv(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    const auto* nk = attrs ? attrs->GetInt(0) : nullptr;
    if (!x || !nk || *nk <= 0 || *nk > std::numeric_limits<int32_t>::max() || x->GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    const int64_t split = x->GetDim(0), group = x->GetDim(3);
    int64_t nq = -1;
    if (split >= 0 && group >= 0) {
        if (!split || *nk % split || group > std::numeric_limits<int64_t>::max() / split) {
            return ge::GRAPH_FAILED;
        }
        nq = split * group - 2 * *nk;
        if (nq <= 0 || nq % *nk) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        auto* out = context->GetOutputShape(i);
        if (!out) return ge::GRAPH_FAILED;
        *out = gert::Shape({x->GetDim(1), x->GetDim(2), i == 0 ? nq : *nk, x->GetDim(4)});
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeUnpermuteQkv(gert::InferDataTypeContext* context)
{
    for (size_t i = 0; i < 3; ++i) {
        if (context->SetOutputDataType(i, context->GetInputDataType(0)) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ops
