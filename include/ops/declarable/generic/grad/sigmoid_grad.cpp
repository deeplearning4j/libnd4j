//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(sigmoid_grad, 1, 1, true, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* applied = nullptr;

            auto z = OUTPUT_VARIABLE(0);
            
            if (block.width() == 2) {
                applied = INPUT_VARIABLE(1);
            } else {
                applied = new NDArray<T>(input);
                input->template applyTransform<simdOps::Sigmoid<T>>(applied, nullptr);
            }

            auto lambda = LAMBDA_TT(_x, _y) {
                return (_x * _y) * ((T) 1.0f -_y);
            };

            input->applyPairwiseLambda(applied, lambda, z);  

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(SigmoidGrad, sigmoid_grad);
    }
}