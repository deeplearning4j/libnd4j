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
            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicEnter<float>;
        template class ND4J_EXPORT LogicEnter<float16>;
        template class ND4J_EXPORT LogicEnter<double>;
    }
}