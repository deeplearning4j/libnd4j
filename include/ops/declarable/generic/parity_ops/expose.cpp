//
// Created by raver119 on 12/11/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(expose, -1, -1, true) {

            for (int e = 0; e < block.width(); e++) {
                auto inVar = block.variable(e);
                if (inVar->variableType() == VariableType::NDARRAY) {
                    auto in = INPUT_VARIABLE(e);
                    auto out = OUTPUT_VARIABLE(e);

                    out->assign(in);
                } else if (inVar->variableType() == VariableType::ARRAY_LIST) {
                    auto list = inVar->getNDArrayList();

                    auto outVar = block.getVariableSpace()->getVariable(block.getNodeId(), e);
                    outVar->setNDArrayList(list);
                }
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Enter, expose);
        DECLARE_SYN(enter, expose);
    }
}