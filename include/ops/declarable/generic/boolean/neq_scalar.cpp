//
// Created by raver119 on 13.10.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

        BOOLEAN_OP_IMPL(neq_scalar, 2, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            if (x->getScalar(0) != y->getScalar(0))
                return ND4J_STATUS_TRUE;
            else
                return ND4J_STATUS_FALSE;
        }
    }
}