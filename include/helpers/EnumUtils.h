//
// @author raver119@gmail.com
//

#ifndef ND4J_ENUM_UTILS_H
#define ND4J_ENUM_UTILS_H

#include <graph/VariableType.h>

namespace nd4j {
    class EnumUtils {
    public:
        static const char * _VariableTypeToString(nd4j::graph::VariableType variableType);
    };
}

#endif