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

            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId())) {
                auto var = varSpace->getVariable(block.getNodeId());
                var->setNDArrayList(list);
            } else {
                auto var = new Variable<T>(nullptr, nullptr, block.getNodeId(), 0);
                var->setNDArrayList(list);
                varSpace->putVariable(block.getNodeId(), var);
            }

            return ND4J_STATUS_OK;
        }
    }
}
