//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicWhile.h>
#include <GraphExecutioner.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicWhile<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            int scopeConditionIndex = node->input()->at(0).first;
            int scopeBodyIndex = node->input()->at(1).first;

            // we're running condition nodes now
            auto scope = graph->scopeById(scopeConditionIndex);
            int breaker = 0;
            while (true && breaker < 20) {
                int lastNode = 0;
                for (auto v: *scope->nodes()) {
                    GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }

                // now we should take result of the Scope run, and evaluate it
                //nd4j_debug("", "");
                auto result = __variableSpace->getVariable(lastNode)->getNDArray();
                result->printBuffer("Result of the last node:");

                // if result evaluates to 0.0 - condition returned FALSE
                if (result->getScalar(0) == (T) 0.0f)
                    break;
                else {
                    auto scopeBody = graph->scopeById(scopeBodyIndex);
                    int lastNode = 0;
                    for (auto v: *scopeBody->nodes()) {
                        GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                        lastNode = v->id();
                    }
                }

                breaker++;
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicWhile<float>;
//        template class ND4J_EXPORT LogicWhile<float16>;
//        template class ND4J_EXPORT LogicWhile<double>;
    }
}
