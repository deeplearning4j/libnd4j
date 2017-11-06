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

            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId())) {
                auto var = varSpace->getVariable(block.getNodeId());
                var->setNDArray(result);
            } else {
                auto var = new Variable<T>(result, nullptr, block.getNodeId(), 0);
                varSpace->putVariable(block.getNodeId(), var);
            }

            return ND4J_STATUS_OK;
        }
    }
}
