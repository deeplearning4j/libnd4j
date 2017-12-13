//
//  @author raver119@gmail.com
//

#include <dll.h>
#include <graph/VariableProxy.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        VariableProxy<T>::VariableProxy(VariableSpace<T>* ref) {
            if (ref == nullptr)
                _backed = new VariableSpace<T>();

            _backed = ref;
        }

        template <typename T>
        bool VariableProxy<T>::hasVariable(int id) {
            return _current.hasVariable(id) || _backed->hasVariable(id);
        }
        
        template <typename T>
        bool VariableProxy<T>::hasVariable(int id, int idx) {
            return _current.hasVariable(id, idx) || _backed->hasVariable(id, idx);
        }
        
        template <typename T>
        bool VariableProxy<T>::hasVariable(std::pair<int,int>& pair) {
            return _current.hasVariable(pair) || _backed->hasVariable(pair);
        }

        template <typename T>
        bool VariableProxy<T>::hasVariable(std::string *symbol) {
            return _current.hasVariable(symbol) || _backed->hasVariable(symbol);
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(int id) {
            return nullptr;
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(int id, int idx) {
            return nullptr;
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(std::pair<int,int>& pair) {
            return nullptr;
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(std::string *symbol) {
            return nullptr;
        }

        template <typename T>
        void VariableProxy<T>::putVariable(std::pair<int,int>& pair, NDArray<T> *array) {
            _current.putVariable(pair, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(std::pair<int,int>& pair, Variable<T> *variable) {
            _current.putVariable(pair, variable);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, Variable<T> *variable) {
            _current.putVariable(id, variable);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, NDArray<T> *array) {
            _current.putVariable(id, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, int idx, NDArray<T> *array) {
            _current.putVariable(id, idx, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, int idx, Variable<T> *array) {
            _current.putVariable(id, idx, array);
        }


        template class ND4J_EXPORT VariableProxy<float>;
        template class ND4J_EXPORT VariableProxy<float16>;
        template class ND4J_EXPORT VariableProxy<double>;
    }
}