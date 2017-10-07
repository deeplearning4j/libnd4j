//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_CUSTOMOPERATIONS_H
#define LIBND4J_CUSTOMOPERATIONS_H

#include <op_boilerplate.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>

namespace nd4j {
    namespace ops {
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);

        DECLARE_OP(noop, -1, -1, true);
        DECLARE_OP(testop2i2o, 2, 2, true);
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);
        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_OP(floor, 1, 1, true);
        DECLARE_OP(realdiv, 2, 1, true);
        DECLARE_OP(merge, -1, 1, true);         // should become custom
        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        DECLARE_OP(assign, 2, 1, false);
        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);
        DECLARE_OP(identity, 1, 1, true);
        DECLARE_OP(add, 2, 1, true);
        DECLARE_OP(subtract, 2, 1, true);
        DECLARE_OP(reversesubtract, 2, 1, true);
        DECLARE_OP(multiply, 2, 1, true);
        DECLARE_OP(divide, 2, 1, true);
        DECLARE_OP(reversedivide, 2, 1, true);
        DECLARE_OP(reshapeas, 2, 1, true);      // should become custom
        DECLARE_OP(transpose, 1, 1, true);      // should become custom



        DECLARE_DIVERGENT_OP(Switch, 2, 2, true);

        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);
        DECLARE_CUSTOM_OP(conv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);
        DECLARE_CUSTOM_OP(reshape, 1, 1, true, 0, -1);

        DECLARE_CONFIGURABLE_OP(tensormmul, 2, 1, false, 0, -1);   // should become custom
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1);
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(repeat, 1, 1, true, 0, -1);   // should become custom
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(permute, 1, 1, true, 0, -1);   // MAYBE should become custom :/
        DECLARE_CONFIGURABLE_OP(sum, 1, 1, false, 0, -1);       // should become reduction
        DECLARE_CONFIGURABLE_OP(batchnorm, 1, 1, true, 4, 3);
    }
}

#endif //LIBND4J_CUSTOMOPERATIONS_H
