//
// Created by raver119 on 21.10.17.
//

#include <pointercast.h>
#include <graph/execution/LogicSwitch.h>
#include <GraphExecutioner.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicSwitch<T>::processNode(Graph<T>* graph, Node<T>* node) {
            auto __variableSpace = graph->getVariableSpace();

            int scopeConditionIndex = node->input()->at(0).first;
            int scopeFalseIndex = node->input()->at(1).first;
            int scopeTrueIndex = node->input()->at(2).first;

            auto scopeCondition = graph->scopeById(scopeConditionIndex);
            int lastNode = 0;
            for (auto v: *scopeCondition->nodes()) {
                GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                lastNode = v->id();
            }

            // now we should take result of the Scope run, and evaluate it
            auto result = __variableSpace->getVariable(lastNode)->getNDArray();
            result->printBuffer("Result of the last node");



            return ND4J_STATUS_OK;
        };

        template class ND4J_EXPORT LogicSwitch<float>;
        //template class ND4J_EXPORT LogicSwitch<float16>;
        //template class ND4J_EXPORT LogicSwitch<double>;
    }
}
