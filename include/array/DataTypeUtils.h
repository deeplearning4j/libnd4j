//
// @author raver119@gmail.com
//

#include <array/DataType.h>
#include <graph/generated/array_generated.h>

namespace nd4j {
    class DataTypeUtils {
    public:
        static DataType fromInt(int dtype);
        static DataType fromFlatDataType(nd4j::graph::DataType dtype);
    };
}