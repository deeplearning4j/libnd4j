//
//  @author raver119@gmail.com
//

#ifndef ND4J_CONTEXT_PROTOTYPE_H
#define ND4J_CONTEXT_PROTOTYPE_H

#include <vector>


namespace nd4j {
    namespace graph {
        template <typename T>
        class ContextPrototype {
        protected:
            // int ids of the input nodes
            std::vector<std::pair<int, int>> _inputs;
            int _nodeId;
            std::vector<T> _tArgs;
            std::vector<int> _iArgs;            
			
			bool _isInplace;

            // opNum for legacy XYZ ops
            int _opNum = -1;

        public:
            ContextPrototype() = default;
            ~ContextPrototype() = default;
        };
    }
}

#endif //ND4J_CONTEXT_PROTOTYPE_H