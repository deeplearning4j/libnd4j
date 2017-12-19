//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/blas.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(batched_gemm, -1, -1, false, -2, 1) {

            return ND4J_STATUS_OK;
        };


        DECLARE_SHAPE_FN(batched_gemm) {

            return new ShapeList();
        }
    }
}