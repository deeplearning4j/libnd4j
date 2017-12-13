//
// Created by raver119 on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation returns index of max element in a given NDArray (optionally: along given dimension(s))
         * Expected input:
         * 0: N-dimensional array
         * 1: optional axis vector
         * 
         * Int args:
         * 0: optional axis
         */
        REDUCTION_OP_IMPL(argmax, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // FIXME: optional vector
            input->template applyIndexReduce<simdOps::IndexMax<T>>(output, *block.getIArguments());

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
    }
}
