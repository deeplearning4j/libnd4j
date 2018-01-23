//
// Created by raver119 on 23.01.18.
//

#ifndef LIBND4J_GRAPHSTATE_H
#define LIBND4J_GRAPHSTATE_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <dll.h>
#include <vector>
#include <map>
#include <Scope.h>
#include <Status.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
namespace graph {
    template <typename T>
    class ND4J_EXPORT GraphState {
    protected:
        // id of this GraphState instance
        Nd4jIndex _id = 0;

        // map of scopes. Scope id is used as key, since it's referred in calls later anyway
        std::map<int, Scope<T>> _scopes;

        // this variable space holds temp references
        VariableSpace<T> _variableSpace;

    public:
        GraphState() = default;
        ~GraphState() = default;


        Nd4jIndex FORCEINLINE id() {
            return _id;
        }

        /**
         * This method adds scope to this state tracker
         *
         * @param scopeId
         * @return
         */
        Nd4jStatus FORCEINLINE registerScope(int scopeId) {
            Scope<T> scope(scopeId);
            _scopes[scopeId] = scope;

            return Status::OK();
        }

        /**
         * This method removes specified scope from this state tracker
         *
         * @param scopeId
         * @return
         */
        Nd4jStatus FORCEINLINE forgetScope(int scopeId) {
            if (_scopes.count(scopeId) > 0)
                _scopes.erase(scopeId);
            else
                return Status::THROW("Non-existent scope requested");

            return Status::OK();
        }

        /**
         * This method adds given op to the end of specified scope
         *
         * @param scopeId
         * @param op
         * @return
         */
        FORCEINLINE Nd4jStatus attachOpToScope(int scopeId, DeclarableOp<T> *op) {

            return Status::OK();
        }

        /**
         * This method returns current variable space of this state holder
         *
         * @return
         */
        FORCEINLINE VariableSpace<T>*  variableSpace() {
            return &_variableSpace;
        }


    };
}
}



#endif //LIBND4J_GRAPHSTATE_H
