//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(rationaltanh, 1, 1, true, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            input->template applyTransform<simdOps::RationalTanh<T>>(output, nullptr);
            STORE_RESULT(output);
            
            return ND4J_STATUS_OK;
        }

        CONFIGURABLE_OP_IMPL(rationaltanh_bp, 2, 1, true, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsilon = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto lambda = LAMBDA_TT(_x, _e) {
                return _e * simdOps::RationalTanhDerivative<T>::op(_x, nullptr);
            };

            input->applyPairwiseLambda(epsilon, lambda, z);  

            return ND4J_STATUS_OK;
        }
    }
}