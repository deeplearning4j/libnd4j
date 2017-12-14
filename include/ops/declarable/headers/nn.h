//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);

        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);

        DECLARE_CUSTOM_OP(batchnorm, 5, 1, false, 1, 2);
    }
}