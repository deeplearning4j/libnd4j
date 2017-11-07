//
// @author raver119@gmail.com
//

#include <array/DataTypeUtils.h>

namespace nd4j {
    DataType DataTypeUtils::fromInt(int val) {
        return (DataType) val;
    }

    DataType DataTypeUtils::fromFlatDataType(nd4j::graph::DataType dtype) {
        return (DataType) dtype;
    }
}