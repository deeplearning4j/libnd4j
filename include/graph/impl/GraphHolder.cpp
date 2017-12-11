//
//  @author raver119@gmail.com
//

#include <graph/GraphHolder.h>

namespace nd4j {
    namespace graph {

        GraphHolder* GraphHolder::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new GraphHolder();

            return _INSTANCE;
        };

        template <typename T>
        void GraphHolder::registerGraph(Nd4jIndex graphId, Graph<T>* graph) {

        }
            
        template <>
        Graph<float>* GraphHolder::pullGraph(Nd4jIndex graphId) {
            if (_graphF.count(graphId) < 1) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw "Bad argument";
            }

            auto graph = _graphF[graphId]->clone();

            return graph;
        }

        template <typename T>
        void GraphHolder::unregisterGraph(Nd4jIndex graphId) {

        }



        GraphHolder* GraphHolder::_INSTANCE = 0;
    }
}