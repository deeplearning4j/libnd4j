//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_CUSTOMOPERATIONS_H
#define LIBND4J_CUSTOMOPERATIONS_H

#include <ops/declarable/headers/activations.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/headers/convo.h>
#include <ops/declarable/headers/list.h>

namespace nd4j {
    namespace ops {
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);
        DECLARE_REDUCTION_OP(argmax, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(argmin, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(norm, 1, 1, false, 1, -2);

        DECLARE_OP(noop, -1, -1, true);
        DECLARE_OP(testop2i2o, 2, 2, true);
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);
        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_OP(floor, 1, 1, true);
        DECLARE_OP(merge, -1, 1, true);         // should become custom
        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);
        DECLARE_OP(zeros_as, 1, 1, false);
        DECLARE_OP(ones_as, 1, 1, false);
        DECLARE_OP(square, 1, 1, true);
        DECLARE_OP(log1p, 2, 1, true);
        DECLARE_OP(toggle_bits, -1, -1, true);
        DECLARE_OP(rint, 1, 1, true);
        DECLARE_OP(listdiff, 2, 2, false);

        DECLARE_OP(scatter_add, 3, 1, true);
        DECLARE_OP(scatter_sub, 3, 1, true);
        DECLARE_OP(scatter_mul, 3, 1, true);
        DECLARE_OP(scatter_div, 3, 1, true);
        DECLARE_OP(scatter_upd, 3, 1, true);
    
        DECLARE_DIVERGENT_OP(Switch, 2, 2, true);

        DECLARE_LOGIC_OP(While);
        DECLARE_LOGIC_OP(Scope);
        DECLARE_LOGIC_OP(Conditional);
        DECLARE_LOGIC_OP(Return);

        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);
        DECLARE_CUSTOM_OP(reshape, 1, 1, true, 0, -2);
        DECLARE_CUSTOM_OP(tear, 1, -1, false, 0, -1);
        DECLARE_CUSTOM_OP(unstack, 1, -1, false, 0, 1);
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, false, 0, 5); // TODO: new op type needed. that returns VIEW
        DECLARE_CUSTOM_OP(slice, 1, 1, false, 0, -1);;
        DECLARE_CUSTOM_OP(tensormmul, 2, 1, false, 0, -1);   
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1);    
        DECLARE_CUSTOM_OP(permute, 1, 1, true, 0, -2);   
        DECLARE_CUSTOM_OP(reshapeas, 2, 1, true, 0, 0);      
        DECLARE_CUSTOM_OP(transpose, 1, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(stack, -1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(size, 1, 1, false, 0, 0); // add DeclarableScalarOp?
        DECLARE_CUSTOM_OP(rank, 1, 1, false, 0, 0); // ^^^^
        DECLARE_CUSTOM_OP(onehot, 1, 1, false, 2, 2);
        DECLARE_CUSTOM_OP(expand_dims, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(range, -2, 1, false, -2, -2);
        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(expose, -1, -1, true, 0, 0);
        DECLARE_CUSTOM_OP(shape_of, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(gather, 2, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(biasadd_bp, 3, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(absoluteDifference, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(cosineDistance, 3, 1, false, 0, 2);
        DECLARE_CUSTOM_OP(squeeze, -1, -1, true, 0, 0);
        DECLARE_CUSTOM_OP(hingeLoss, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(huberLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(logLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(meanPairWsSqErr, 3, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(meanSqErr, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(sigmCrossEntropy, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(softmaxCrossEntropy, 3, 1, false, 1, 1);      
        DECLARE_CUSTOM_OP(batchnorm, 5, 1, false, 1, 2);
        DECLARE_CUSTOM_OP(unique, 1, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(lstmCell, 8, 2, false, 3, 2);
        DECLARE_CUSTOM_OP(set_seed, -2, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(get_seed, -2, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(shapes_of, -1, -1, false, 0, 0)
        DECLARE_CUSTOM_OP(sruCell, 4, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(gruCell, 5, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(diag, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(diag_part, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(tile, 1, 1, false, 0, -2);

        // recurrent ops
        DECLARE_CUSTOM_OP(sru,         5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru_logic,   5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru_bi,      5, 2, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bp,      8, 4, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bp_logic,8, 4, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bi_bp,   8, 4, true,  0, 0);
                
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1);        
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0);        
        // DECLARE_CONFIGURABLE_OP(batchnorm_bp, 5, 1, true, 0, 1);                
        
        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(axpy, 2, 1, false, -2, 0);
        DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);        
        DECLARE_CONFIGURABLE_OP(matrix_set_diag, 2, 1, false, 0, 0)
        DECLARE_CONFIGURABLE_OP(betainc, 3, 1, false, 0, 0)

        DECLARE_CUSTOM_OP(firas_sparse, 1, 1, false, 0, -1);

    }
}

#endif //LIBND4J_CUSTOMOPERATIONS_H
