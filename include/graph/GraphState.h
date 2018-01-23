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


    };
}
}



#endif //LIBND4J_GRAPHSTATE_H
