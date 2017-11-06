//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(size_list, 1, 1, 0, 0) {
            auto list = INPUT_LIST(0);

            auto result = NDArrayFactory<T>::scalar((T) list->height());

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