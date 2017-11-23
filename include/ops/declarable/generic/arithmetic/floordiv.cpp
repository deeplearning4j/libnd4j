//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(floordiv, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                // REQUIRE_OK(this->validateInputDimensionsMatch(block));
                x->template applyPairwiseTransform<simdOps::FloorDiv<T>>(y, z, nullptr);

            } else if (!x->isScalar() && y->isScalar()) {
                x->template applyScalar<simdOps::FloorDiv<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::FloorDiv<T>>(*z, y);

            }
            else if (x->isScalar() && y->isScalar()) { // (x->isScalar() && y->isScalar())
                z->putScalar(0, simdOps::FloorDiv<T>::op(x->getScalar(0),y->getScalar(0), nullptr));
            } else {
                auto tZ = x->template applyTrueBroadcast<simdOps::FloorDiv<T>>(y);
                OVERWRITE_RESULT(tZ);
            }

            return ND4J_STATUS_OK;
        }
    }
}