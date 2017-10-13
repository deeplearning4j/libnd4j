//
// Created by raver119 on 13.10.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /*
         * arg_0 is our "signal"
         * arg_1 is condition that will determine transition
         */
        DIVERGENT_OP_IMPL(Switch, 2, 2, true) {
            auto input = INPUT_VARIABLE(0);
            auto condition = INPUT_VARIABLE(1);

            // we'll store signal to both ends
            STORE_2_RESULTS(*input, *input);

            // but we'll ensure only one node is active, and other is disabled
            if (condition->getScalar(0) == (T) 0.0f)
                block.setBranch(0);
            else
                block.setBranch(1);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(switch, Switch);
        DECLARE_SYN(if, Switch);
    }
}