//
//  @author raver119@gmail.com
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(add, 2, 1, true, 0, 0) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Add<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

    		return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(add) {
            auto shapeList = new ShapeList();
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
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape);

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

        CUSTOM_OP_IMPL(add_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto eps = OUTPUT_VARIABLE(0);
            auto grad = OUTPUT_VARIABLE(1);

            if (x->isSameShape(y)) {
                // PWT case case
                grad->assign(epsNext);
                eps->assign(epsNext);
            } else if (y->isScalar()) {
                // scalar case
                auto tmp = epsNext->template reduceNumber<simdOps::Sum<T>>();
                grad->assign(tmp);
                eps->assign(epsNext);
            } else {
                // broadcast case
                auto sum = epsNext->template reduceAlongDimension<simdOps::Sum<T>>({1});
                grad->assign(sum);
                delete sum;
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(add_bp) {
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);
            auto e = inputShape->at(2);

            // eps always has shape of x
            // grad always has shape of y

            int *shapeE;
            int *shapeG;
            ALLOCATE(shapeE, block.getWorkspace(), shape::shapeInfoLength(x), int);
            ALLOCATE(shapeG, block.getWorkspace(), shape::shapeInfoLength(y), int);

            REPLICATE_SHAPE(x, shapeE);
            REPLICATE_SHAPE(y, shapeG);

            auto shapeList = new ShapeList({shapeE, shapeG});

            return shapeList;
        }
    }
}