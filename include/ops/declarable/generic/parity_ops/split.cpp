//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(split, 1, -1, false, 0, 1) {
        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(split) {
        return new ShapeList();
    }
}
}