//
// Created by raver119 on 01/11/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(onehot, 1, 1, false, 2, 2) {
            auto input = INPUT_VARIABLE(0);

            T on = T_ARG(0);
            T off = T_ARG(1);

            auto depth = INT_ARG(0);
            auto axis = INT_ARG(0);

            REQUIRE_TRUE(input->isVector(), 0, "One-hot input should be Vector, but got %iD instead", input->rankOf());

            auto output = OUTPUT_VARIABLE(0);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(onehot) {
            int *inShape = inputShape->at(0);

            auto depth = INT_ARG(0);
            auto axis = INT_ARG(0);

            return new ShapeList();
        }
    }
}
