//
// Created by raver119 on 23.01.18.
//

#include <graph/GraphState.h>
#include <graph/Node.h>


namespace nd4j {
namespace graph {

    template <typename T>
    GraphState<T>::GraphState(Nd4jIndex id) {
        _id = id;
    };

    template <typename T>
    GraphState<T>::~GraphState() {
        // stupid. we should get rid of pointers here i think. no need to bother
        for (const auto &v: _scopes) {
            delete v.second;
        }
    };

    template <typename T>
    Nd4jStatus GraphState<T>::registerScope(int scopeId) {
        auto scope = new Scope<T>(scopeId);
        _scopes[scopeId] = scope;

        return Status::OK();
    };

    template <typename T>
    Nd4jStatus GraphState<T>::forgetScope(int scopeId) {
        if (_scopes.count(scopeId) > 0)
            _scopes.erase(scopeId);
        else
            return Status::THROW("Non-existent scope requested");

        return Status::OK();
    };

    template <typename T>
    Nd4jStatus GraphState<T>::attachOpToScope(int scopeId, DeclarableOp<T> *op, std::vector<Pair> inputs) {

        return Status::OK();
    };

    template <typename T>
    VariableSpace<T>* GraphState<T>::variableSpace() {
        return &_variableSpace;
    };

    template <typename T>
    Nd4jIndex GraphState<T>::id() {
        return _id;
    }

    template <typename T>
    Nd4jStatus GraphState<T>::attachOpToScope(int scopeId, Nd4jIndex opNum, OpType type, std::vector<Pair> inputs) {
        // we should use OpRegistrator here, to create Node and push it to specific scope
        return Status::OK();
    }

    template class ND4J_EXPORT GraphState<float>;
    template class ND4J_EXPORT GraphState<double>;
    template class ND4J_EXPORT GraphState<float16>;
}
}