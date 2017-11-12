//
// Created by raver119 on 12.11.2017.
//

#include <graph/execution/LogicExpose.h>

namespace nd4j {
    namespace graph {
        template<typename T>
        Nd4jStatus LogicExpose<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // do we really want this?
            return ND4J_STATUS_OK;
        }
    }
}