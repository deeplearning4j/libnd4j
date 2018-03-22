//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(less, 2, 1, true, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y) || (x->isScalar() && y->isScalar()) || (x->isVector() && y->isVector()), 0, "Both inputs should have equal shape");

            auto lambda = LAMBDA_TT(_x, _y) {
                return _x < _y ? (T) 1.0f : (T) 0.0f;
            };

            x->applyPairwiseLambda(y, lambda, z);

            STORE_RESULT(z);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(less) {
            auto shapeList = SHAPELIST();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);

            if (shape::equalsSoft(x, y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(y), int);
                REPLICATE_SHAPE(y, newshape);

                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (ShapeUtils<T>::areShapesBroadcastable(x, y)) {
                int *newshape = nullptr;
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                int *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(x), int);
                REPLICATE_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        }
    }
}