//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(read_list, 1, 1, 0, 1) {
            auto list = INPUT_LIST(0);

            auto index = INT_ARG(0);
            auto result = list->read(index);

            OVERWRITE_RESULT(result);

            return ND4J_STATUS_OK;
        }
    }
}
