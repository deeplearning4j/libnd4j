//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(space_to_batch, 3, 1, false, 0, -2) {

        return Status::OK();
    }

    DECLARE_SHAPE_FN(space_to_batch) {
        return new ShapeList();
    }
}
}