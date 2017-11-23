//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(realdiv, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);

            if (!x->isScalar() && !y->isScalar()) {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                // REQUIRE_OK(this->validateInputDimensionsMatch(block));
                x->template applyPairwiseTransform<simdOps::Divide<T>>(y, z, nullptr);

            } else if (!x->isScalar() && y->isScalar()) {
                x->template applyScalar<simdOps::Divide<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Divide<T>>(*x, z);
            }
            else if (x->isScalar() && y->isScalar()) { // (x->isScalar() && y->isScalar())
                z->putScalar(0, x->getScalar(0) / y->getScalar(0));
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RealDiv, realdiv);
    }
}