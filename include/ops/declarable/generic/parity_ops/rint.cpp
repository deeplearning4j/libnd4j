//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation applies element-wise rint (round to integral value) operation
         */
        OP_IMPL(rint, 1, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            x->template applyTransform<simdOps::Rint<T>>(z);

            return ND4J_STATUS_OK;
        }
    }
}