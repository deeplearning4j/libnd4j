//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICWHILE_H
#define LIBND4J_LOGICWHILE_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class LogicWhile {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICWHILE_H
