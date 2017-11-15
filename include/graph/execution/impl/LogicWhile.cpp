//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicWhile.h>
#include <graph/execution/LogicReturn.h>
#include <GraphExecutioner.h>
#include <graph/execution/LogicExecutor.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicWhile<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            nd4j_debug("Starting on WHILE loop: [%i]\n", node->id());

            // total number of inputs. 2 last inputs are scopes
            int inputs = node->input()->size();

            if (inputs < 3) {
                nd4j_printf("While [%i]: loop should have at least 1 external variable announced\n", node->id());
                return ND4J_STATUS_BAD_INPUT;
            }

            for (int e = 0; e < inputs - 2; e++) {
                std::pair<int, int> pair(node->id(), e);
                if (!__variableSpace->hasVariable(pair)) {
                    __variableSpace->putVariable(pair, new Variable<T>(nullptr, nullptr, node->id(), e));
                }

                auto inputVar = __variableSpace->getVariable(node->input()->at(e));

                auto innerVar = __variableSpace->getVariable(pair);
                if (innerVar->hasNDArray()) {
                    // TODO: ???
                } else {
                    // FIXME: in some cases it's possible to have no NDArray
                    if (inputVar->hasNDArray())
                        innerVar->setNDArray(inputVar->getNDArray()->dup());
                }
            }

            int scopeConditionIndex = node->input()->at(inputs - 2).first;
            int scopeBodyIndex = node->input()->at(inputs - 1).first;

            nd4j_debug("While [%i]: got [%i] inputs\n", node->id(), node->input()->size());

            // we're running condition nodes now
            auto scope = graph->scopeById(scopeConditionIndex);
            int breaker = 0;
            while (true && breaker < 10000000) {
                int lastNode = 0;
                // we're running condition scope first
                nd4j_debug("While [%i]: got [%i] ops in condition scope [%i]\n", node->id(), scope->nodes()->size(), scopeConditionIndex);

                for (Node<T>* v: *scope->nodes()) {
                    //v->getBlock()->updateVariables();
                    if (v->opType() == OpType_LOGIC) {
                        nd4j_debug("Falling back to logic\n","");
                        LogicExecutor<T>::processNode(graph, v);
                    } else {
                        nd4j_debug("Op [<%s>]\n", v->getName()->c_str());
                        Nd4jStatus status = GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                        if (status != ND4J_STATUS_OK)
                            return status;
                    }

                    lastNode = v->id();
                }

                if (!__variableSpace->hasVariable(lastNode)) {
                    nd4j_printf("While [%i]: got no results out of conditional loop\n", node->id());
                    return ND4J_STATUS_KERNEL_FAILURE;
                }

                // now we should take result of the Scope run, and evaluate it
                auto result = __variableSpace->getVariable(lastNode)->getNDArray();

                if (Environment::getInstance()->isDebugAndVerbose())
                    result->printBuffer("Result of the last node:");

                // if result evaluates to 0.0 - condition returned FALSE
                if (result->getScalar(0) == (T) 0.0f)
                    break;
                else {
                    auto scopeBody = graph->scopeById(scopeBodyIndex);
                    int lastNode = 0;
                    int e = 0;
                    nd4j_debug("While [%i] got [%i] ops in body scope [%i]\n", node->id(), scopeBody->nodes()->size(), scopeBodyIndex);
                    for (; e < scopeBody->nodes()->size() - 1; e++) {
                        Node<T>* v = scopeBody->nodes()->at(e);

                        if (v->opType() == OpType_LOGIC) {
                            nd4j_debug("Falling back to logic\n","");
                            LogicExecutor<T>::processNode(graph, v);
                        } else {
                            nd4j_debug("Op [<%s>]\n", v->getName()->c_str());
                            //v->getBlock()->updateVariables();
                            Nd4jStatus status = GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                            if (status != ND4J_STATUS_OK)
                                return status;
                        }

                        lastNode = v->id();
                    }

                    // now execute return statement
                    Node<T>* ret = scopeBody->nodes()->at(e);
                    LogicReturn<T>::processNode(graph, ret);
                }

                breaker++;
            }

            // if we've hit breaker limit - we should notify about that
            if (breaker >= 10000000) {
                nd4j_printf("While condition seems to be never ending, aborting...\n",  breaker);
                return ND4J_STATUS_KERNEL_FAILURE;
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicWhile<float>;
//        template class ND4J_EXPORT LogicWhile<float16>;
//        template class ND4J_EXPORT LogicWhile<double>;
    }
}
