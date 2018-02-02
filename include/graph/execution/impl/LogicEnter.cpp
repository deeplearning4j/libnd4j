//
//  @author raver119@gmail.com
//

#include <graph/execution/LogicEnter.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicEnter<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // this op is basically no-op
            // we just know it exists

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

//                    if (lvar->hasNDArray())
//                        delete lvar->getNDArray();

                    auto array = var->getNDArray();
                    lvar->setNDArray(array);
                    lvar->markReadOnly(true);
                    //lvar->markExternal(false);h

                    break;
                }
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicEnter<float>;
        template class ND4J_EXPORT LogicEnter<float16>;
        template class ND4J_EXPORT LogicEnter<double>;
    }
}