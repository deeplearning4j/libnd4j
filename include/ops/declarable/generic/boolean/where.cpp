//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(where, 1, 1, false, 0, 0) {
            auto condition = INPUT_VARIABLE(0);

            if (block.width() == 3) {
                auto x = INPUT_VARIABLE(1);
                auto y = INPUT_VARIABLE(2);

                auto z = OUTPUT_VARIABLE(0);

                REQUIRE_TRUE(x->isSameShape(y), 0, "X and Y must have equal shapes");
                if (condition->isSameShape(x)) {
                    // FIXME: for perf it should be better to issue memcpy here, and fill only mismatched values from either X or Y
#pragma omp parallel for
                    for (int e = 0; e < condition->lengthOf(); e++) {
                        T v = condition->getIndexedScalar(e);
                        T r = v == (T) 0.0f ? y->getIndexedScalar(e) : x->getIndexedScalar(e);
                        z->putIndexedScalar(e, r);
                    }
                }
            } else {

            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(where) {
            if (block.width() == 3) {
                auto inShape = inputShape->at(1);
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
                memcpy(newshape, inShape, shape::shapeInfoByteLength(inShape));

                return new ShapeList(newshape);
            } else {

                // FIXME: we can't estimate result here in this case
                return nullptr;
            }
        }
    }
}