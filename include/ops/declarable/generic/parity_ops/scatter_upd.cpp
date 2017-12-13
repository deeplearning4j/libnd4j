//
// Created by raver119 on 24.11.17.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation applies Assign opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        OP_IMPL(scatter_upd, 3, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto indices = INPUT_VARIABLE(1);
            auto updates = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            ScatterHelper<T>::template scatter_apply<simdOps::Copy<T>>(output, indices, updates);        

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ScatterUpdate, scatter_upd);
    }
}