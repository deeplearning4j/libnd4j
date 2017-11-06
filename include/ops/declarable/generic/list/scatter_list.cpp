//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(scatter_list, 1, 1, 0, -1) {

            return ND4J_STATUS_OK;
        }
    }
}