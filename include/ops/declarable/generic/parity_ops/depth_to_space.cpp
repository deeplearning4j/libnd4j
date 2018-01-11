//
// @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(depth_to_space, 1, 1, false, 0, -2) {
        return ND4J_STATUS_OK;
    }
    

    DECLARE_SHAPE_FN(depth_to_space) {
        return nullptr;
    }
}
}