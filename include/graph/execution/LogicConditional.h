//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICCONDITIONAL_H
#define LIBND4J_LOGICCONDITIONAL_H

#include <pointercast.h>
#include <Node.h>
#include <Graph.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class LogicConditional {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICCONDITIONAL_H
