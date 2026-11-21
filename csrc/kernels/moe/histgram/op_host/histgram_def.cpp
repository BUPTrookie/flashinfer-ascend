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
 * \file histgram_def.cpp
 * \brief Histgram 算子原型定义
 *
 * 输入:
 *   input : [numToken]，一维 token 序列，int32，取值属于 [0, numExpert)
 * 属性:
 *   numExpert : int，expert 个数
 * 输出:
 *   out   : [numExpert]，每个 expert 的出现次数，int32
 */
#include <cstdint>
#include <register/op_def_registry.h>
#include "histgram_infershape.cpp"
#include "histgram_tiling.cpp"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {

class FiHistgram : public OpDef {
public:
    explicit FiHistgram(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("counts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_expert").AttrType(REQUIRED).Int();

        SetInferShape(InferShapeHistgram).SetInferDataType(InferDataTypeHistgram);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        AICore().SetTiling(optiling::TilingHistgram).AddConfig(FI_STRING(FI_COMPUTE_UNIT), config);
    }
};

OP_ADD(FiHistgram);
} // namespace ops
