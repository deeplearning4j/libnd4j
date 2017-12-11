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

        template <>
        void GraphHolder::registerGraph(Nd4jIndex graphId, Graph<float>* graph) {
            _graphF[graphId] = graph;
        }
            
        template <>
        Graph<float>* GraphHolder::pullGraph(Nd4jIndex graphId) {
            if (!this->hasGraph<float>(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw "Bad argument";
            }

            auto graph = _graphF[graphId]->clone();

            return graph;
        }

        template <typename T>
        void GraphHolder::forgetGraph(Nd4jIndex graphId) {
            if (sizeof(T) == 4) {
                if (this->hasGraph<float>(graphId))
                    _graphF.erase(graphId);
            }
        }

        template<typename T>
        bool GraphHolder::hasGraph(Nd4jIndex graphId) {
            return _graphF.count(graphId) > 0;
        }


        template bool GraphHolder::hasGraph<float>(Nd4jIndex graphId);
        template void GraphHolder::forgetGraph<float>(Nd4jIndex graphId);


        GraphHolder* GraphHolder::_INSTANCE = 0;
    }
}