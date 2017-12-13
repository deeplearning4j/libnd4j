//
//  @author raver119@gmail.com
//

#include <graph/VariableSpace.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class VariableProxy: public VariableSpace<T> {
        protected:
            VariableSpace<T>* _backed;
            VariableSpace<T> _current;
        public:
            VariableProxy(VariableSpace<T>* reference);
            ~VariableProxy() = default;

            bool hasVariable(int id);
            bool hasVariable(int id, int idx);
            bool hasVariable(std::pair<int,int>& pair);
            bool hasVariable(std::string *symbol);

            nd4j::graph::Variable<T> *getVariable(int id);
            nd4j::graph::Variable<T> *getVariable(int id, int idx);
            nd4j::graph::Variable<T> *getVariable(std::pair<int,int>& pair);
            nd4j::graph::Variable<T> *getVariable(std::string *symbol);

            void putVariable(std::pair<int,int>& pair, NDArray<T> *array);
            void putVariable(std::pair<int,int>& pair, Variable<T> *variable);
            void putVariable(int id, Variable<T> *variable);
            void putVariable(int id, NDArray<T> *array);
            void putVariable(int id, int idx, NDArray<T> *array);
            void putVariable(int id, int idx, Variable<T> *array);
        };
    }
}