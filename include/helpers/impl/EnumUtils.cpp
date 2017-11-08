//
// @author raver119@gmail.com
//

#include <graph/VariableType.h>
#include <helpers/EnumUtils.h>

using namespace nd4j::graph;

namespace nd4j {
    const char * EnumUtils::_VariableTypeToString(nd4j::graph::VariableType variableType) {
        switch (variableType) {
            case NDARRAY: return "NDARRAY";
            case ARRAY_LIST: return "ARRAY_LIST";
            case FLOW: return "FLOW";
            default: return "UNKNOWN VariableType";
        }
    }
}