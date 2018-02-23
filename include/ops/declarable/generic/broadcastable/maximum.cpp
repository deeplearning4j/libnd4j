//
// Created by raver119 on 12.10.2017.
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(maximum, 2, 1, true, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Max<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(maximum) {
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

        CUSTOM_OP_IMPL(maximum_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);

            auto lambdaX = LAMBDA_TTT(_e, _x, _y) {
                return _x >= _y ? _e : (T) 0.;
            };

            auto lambdaY = LAMBDA_TTT(_e, _x, _y) {
                return _x <= _y ? _e : (T) 0.;
            };


            if (x->isSameShape(y)) {
                // PWT case case

                // X gradient
                epsNext->applyTriplewiseLambda(x, y, lambdaX, gradX);

                // Y gradient
                epsNext->applyTriplewiseLambda(x, y, lambdaY, gradY);

            } else if (y->isScalar()) {
                T s = y->getScalar(0);
                auto lambdaS = LAMBDA_TT(_e, _x, s) {
                    return _x >= s ? _e : (T) 0.;
                };

                // scalar case
                T tmp = epsNext->template reduceNumber<simdOps::Sum<T>>();
                gradY->assign( x <= y ? tmp : (T) 0.0f);
                
                epsNext->applyPairwiseLambda(x, lambdaS, gradX);
            } else {
                // broadcast case

                // in this case we want to boost our X and Y shapes to the size of FF pass output (or epsNext, which has the same shape)
                auto preX = x->dup();
                auto preY = y->dup();

                auto targetShape = epsNext->getShapeAsVector();

                preX->tileToShape(targetShape);
                preY->tileToShape(targetShape);

                epsNext->applyTriplewiseLambda(preX, preY, lambdaX, preX);
                epsNext->applyTriplewiseLambda(preX, preY, lambdaY, preY);

                auto axisX = ShapeUtils<T>::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
                auto axisY = ShapeUtils<T>::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

                if (axisX.size() > 0) {
                    auto sum = preX->template reduceAlongDimension<simdOps::Sum<T>>(axisX);
                    gradX->assign(sum);
                    delete sum;
                } else 
                    gradX->assign(preX);

                if (axisY.size() > 0) {
                    auto sum = preY->template reduceAlongDimension<simdOps::Sum<T>>(axisY);
                    gradY->assign(sum);
                    delete sum;
                } else
                    gradY->assign(preY);


                delete preX;
                delete preY;
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(maximum_bp) {
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