//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(create_list, 1, 1, 0, -1) {
            int height = 0;
            bool expandable = false;
            if (block.getIArguments()->size() == 1) {
                height = INT_ARG(0);
                expandable = (bool) INT_ARG(1);
            } else if (block.getIArguments()->size() == 2) {
                height = INT_ARG(0);
            }

            auto list = new NDArrayList<T>(height, expandable);

            // we recieve input array for graph integrity purposes only
            auto input = INPUT_VARIABLE(0);

            OVERWRITE_RESULT(list);

            return ND4J_STATUS_OK;
        }
    }
}