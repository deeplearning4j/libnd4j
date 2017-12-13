//
// Created by raver119 on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation takes input's shape, and returns new NDArray filled with specified value
         * Expected arguments:
         * input: N-dimensional array
         * 
         * T args:
         * 0: scalar value, used to fill NDArray
         */
        CONFIGURABLE_OP_IMPL(fill_as, 1, 1, true, 1, 0) {
            auto output = OUTPUT_VARIABLE(0);
            auto scalr = T_ARG(0);

            output->assign(scalr);

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(filllike, fill_as);
        DECLARE_SYN(fill_like, fill_as);
    }
}