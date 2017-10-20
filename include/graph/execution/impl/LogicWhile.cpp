//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicWhile.h>
#include <GraphExecutioner.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicWhile<T>::processNode(Graph<T> *graph, Node<T> *node) {

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicWhile<float>;
//        template class ND4J_EXPORT LogicWhile<float16>;
//        template class ND4J_EXPORT LogicWhile<double>;
    }
}
