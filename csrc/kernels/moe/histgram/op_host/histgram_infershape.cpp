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
 * \file histgram_infershape.cpp
 * \brief Histgram 的 shape / dtype 推导
 */
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
ge::graphStatus InferShapeHistgram(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    const auto* numExpert = attrs ? attrs->GetInt(0) : nullptr;
    if (!numExpert || *numExpert < 1 || *numExpert > (1 << 24)) return GRAPH_FAILED;
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = gert::Shape({*numExpert});
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeHistgram(gert::InferDataTypeContext* context)
{
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
} // namespace

} // namespace ops
