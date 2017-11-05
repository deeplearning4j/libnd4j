//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(write_list, 2, 0, 0, 0) {
            auto list = INPUT_LIST(0);

            return ND4J_STATUS_OK;
        }
    }
}