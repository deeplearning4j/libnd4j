//
//  @author raver119@gmail.com
//

#include <graph/execution/LogicLoopCond.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicLoopCond<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            Context<T> ctx(node->getContextPrototype(), __variableSpace);
            auto input = ctx.variable(0)->getNDArray();

            std::pair<int, int> pair0(node->id(), 0);

            if (!__variableSpace->hasVariable(pair0))
                __variableSpace->putVariable(pair0, new Variable<T>(nullptr, nullptr, node->id(), 0));

            __variableSpace->getVariable(pair0)->setNDArray(input);
            __variableSpace->getVariable(pair0)->markRemovable(false);

            // pass further
            if (input->getScalar(0) > 0.0) {
                // if condition is TRUE body will be invoked some time soon
                __flowPath->markFrameActive(node->getFrameId(), true);
                //__flowPath->i
            } else {
                // body won't be activated
                __flowPath->markFrameActive(node->getFrameId(), false);
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicLoopCond<float>;
        template class ND4J_EXPORT LogicLoopCond<float16>;
        template class ND4J_EXPORT LogicLoopCond<double>;
    }
}