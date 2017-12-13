//
// Created by raver119 on 12.10.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation takes input's shape, and returns new NDArray filled with zeros
         * Expected arguments:
         * input: N-dimensional array
         * 
         */
        OP_IMPL(zeros_as, 1, 1, false) {
            auto input = INPUT_VARIABLE(0);

            auto out = OUTPUT_VARIABLE(0);

            STORE_RESULT(*out);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(zeroslike, zeros_as);
        DECLARE_SYN(zeros_like, zeros_as);
    }
}

