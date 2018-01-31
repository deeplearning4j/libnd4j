//
// Created by raver119 on 30.01.18.
//

#include <graph/execution/LogicMerge.h>
#include <Status.h>

namespace nd4j {
    namespace graph {
        template<typename T>
        Nd4jStatus LogicMerge<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // at merge node only one of inputs exist (?)
            auto __variableSpace = graph->getVariableSpace();

            // basically, first non-null variable is our target
            for (int e = 0; e < node->input()->size(); e++) {
                auto inputAddr = node->input()->at(e);

                if (__variableSpace->hasVariable(inputAddr)) {
                    auto var = __variableSpace->getVariable(inputAddr);
                    if (!var->hasNDArray())
                        continue;

                    Variable<T> *lvar = nullptr;
                    if (__variableSpace->hasVariable(node->id(), 0))
                        lvar = __variableSpace->getVariable(node->id(), 0);
                    else
                        lvar = new Variable<T>(nullptr, node->getName()->c_str(), node->id(), 0);

                    if (lvar->hasNDArray())
                        delete lvar->getNDArray();

                    auto array = var->getNDArray();
                    lvar->setNDArray(array);
                    lvar->markReadOnly(true);
                    //lvar->markExternal(false);h

                    break;
                }
            }

            return Status::OK();
        }


        template class ND4J_EXPORT LogicMerge<float>;
        template class ND4J_EXPORT LogicMerge<float16>;
        template class ND4J_EXPORT LogicMerge<double>;
    }
}
