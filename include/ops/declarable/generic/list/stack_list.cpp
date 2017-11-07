//
//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(stack_list, 1, 1, 0, 1) {
            auto list = INPUT_LIST(0);
            //auto z = OUTPUT_VARIABLE(0);

            // FIXME: this is obviously bad
            auto result = list->stack();

            OVERWRITE_RESULT(result);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayConcatV3, stack_list);
    }
}