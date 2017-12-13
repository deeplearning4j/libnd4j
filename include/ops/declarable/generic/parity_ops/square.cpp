//
// Created by raver119 on 01/11/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation applies element-wise pow(x, 2) to the given input
         * Expected arguments:
         * input: N-Dimensional array
         */
        OP_IMPL(square, 1, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            T extras = (T) 2.0f;
            input->template applyTransform<simdOps::Pow<T>>(output, &extras);

            return ND4J_STATUS_OK;
        }
    }
}