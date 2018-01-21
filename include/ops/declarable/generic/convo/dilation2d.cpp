//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/convo.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dilation2d, 4, 1, false, 0, 1) {

        return Status::OK();
    }

    DECLARE_SHAPE_FN(dilation2d) {

        return new ShapeList();
    }
}
}