//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
        
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);

        
        DECLARE_CUSTOM_OP(tensormmul, 2, 1, false, 0, -1);   


        DECLARE_CONFIGURABLE_OP(axpy, 2, 1, false, -2, 0);
    }
}