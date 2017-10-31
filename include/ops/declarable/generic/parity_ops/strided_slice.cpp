//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

        CUSTOM_OP_IMPL(strided_slice, 1, 1, true, 0, -1) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(block.getIArguments()->size() == x->rankOf() * 3, 0, "Number of Integer arguments should be equal to input rank x 3, but got %i instead", block.getIArguments()->size());

            STORE_RESULT(z);
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(stridedslice, strided_slice);

        DECLARE_SHAPE_FN(strided_slice) {

            return new ShapeList();
        }
    }
}