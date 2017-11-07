//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(gather_list, 2, 1, 0, -1) {
            auto list = INPUT_LIST(0);
            auto indices = INPUT_VARIABLE(0);

            // first of all we need to get shapes
            for (int e = 0; e < list->height(); e++) {
                
            }

            return ND4J_STATUS_OK;
        }
    }
}