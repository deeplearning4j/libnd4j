#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        BOOLEAN_OP_IMPL(is_non_decreasing, 1, true) {

            auto input = INPUT_VARIABLE(0);
            auto length = shape::length(input->getShapeInfo());

            bool isNonDecreasing = true;

            if(length > 1) {
                for (int i = 0; i < length - 1; i++) {
                    auto elem1 = (*input)(i);
                    auto elem2 = (*input)(i+1);
                    if (elem1 > elem2)
                        isNonDecreasing = false;
                }
            }

            if (isNonDecreasing)
                return ND4J_STATUS_TRUE;
            else
                return ND4J_STATUS_FALSE;
        }
    }
}