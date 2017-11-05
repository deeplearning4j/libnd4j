//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(write_list, 2, 0, 0, 1) {
            auto list = INPUT_LIST(0);
            auto input = INPUT_VARIABLE(1);
            auto idx = INT_ARG(0);

            Nd4jStatus result = list->write(idx, input);

            return result;
        }
    }
}