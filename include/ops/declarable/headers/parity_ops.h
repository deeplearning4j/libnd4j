//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_REDUCTION_OP(argmax, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(argmin, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(norm, 1, 1, false, 1, -2);
        DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);      
        DECLARE_CONFIGURABLE_OP(matrix_set_diag, 2, 1, false, 0, 0);
        DECLARE_CONFIGURABLE_OP(betainc, 3, 1, false, 0, 0);

        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_CUSTOM_OP(biasadd_bp, 3, 2, false, 0, 0);

        DECLARE_CUSTOM_OP(diag, 1, 1, false, 0, 0);

        DECLARE_CUSTOM_OP(diag_part, 1, 1, false, 0, 0);

        DECLARE_OP(listdiff, 2, 2, false);

        DECLARE_OP(scatter_add, 3, 1, true);
        DECLARE_OP(scatter_sub, 3, 1, true);
        DECLARE_OP(scatter_mul, 3, 1, true);
        DECLARE_OP(scatter_div, 3, 1, true);
        DECLARE_OP(scatter_upd, 3, 1, true);

        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 1, 0);

        DECLARE_OP(rint, 1, 1, true);

        DECLARE_CUSTOM_OP(unique, 1, 2, false, 0, 0);

        DECLARE_CUSTOM_OP(tear, 1, -1, false, 0, -1);
        DECLARE_CUSTOM_OP(unstack, 1, -1, false, 0, 1);
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, false, 0, 5); // TODO: new op type needed. that returns VIEW
        DECLARE_CUSTOM_OP(slice, 1, 1, false, 0, -1);

        DECLARE_CUSTOM_OP(range, -2, 1, false, -2, -2);

        DECLARE_CUSTOM_OP(onehot, 1, 1, false, 2, 2);

        DECLARE_CUSTOM_OP(stack, -1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(size, 1, 1, false, 0, 0); // add DeclarableScalarOp?
        DECLARE_CUSTOM_OP(rank, 1, 1, false, 0, 0); // ^


        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        DECLARE_OP(zeros_as, 1, 1, false);
        DECLARE_OP(ones_as, 1, 1, false);
        DECLARE_OP(square, 1, 1, true);
    }
}