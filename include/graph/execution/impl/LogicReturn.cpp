//
// Created by raver119 on 28.10.2017.
//

#include "graph/execution/LogicReturn.h"

namespace nd4j {
    namespace graph {

        template <typename T>
        Nd4jStatus LogicReturn<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            for (int e = 0; e < node->input()->size(); e++) {
                auto inputAddr = node->input()->at(e);
                auto outputAddr = node->output()->at(e);
                auto varIn = __variableSpace->getVariable(inputAddr);
                auto varOut = __variableSpace->getVariable(inputAddr);

                // FIXME: this is obviously wrong, we should keep depth track for backprop here
                varIn->getNDArray()->assign(varOut->getNDArray());
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicReturn<float>;
    }
}
