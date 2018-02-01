//
//  @author raver119@gmail.com
//

#include <graph/execution/LogicNextIteration.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicNextIeration<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();



            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicNextIeration<float>;
        template class ND4J_EXPORT LogicNextIeration<float16>;
        template class ND4J_EXPORT LogicNextIeration<double>;
    }
}