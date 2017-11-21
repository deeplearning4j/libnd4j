//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(crelu, 1, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);

            auto tmp = x->dup();
            tmp->template applyTransform<simdOps::Neg<T>>();

            auto z = OUTPUT_VARIABLE(0);

            NDArrayFactory<T>::concat({x, tmp}, -1, z);

            // TODO: make this configurable?
            T threshold = (T) 0.0f;
            z->template applyTransform<simdOps::RELU<T>>(&threshold);

            STORE_RESULT(z);

            delete tmp;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(crelu) {
            auto inShape = inputShape->at(0);
            std::vector<int> shape;
            for (int e = 0; e < shape::rank(inShape); e++)
                shape.emplace_back(shape::shapeOf(inShape)[e]);
            
            shape[shape.size()-1] *= 2;
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            if (shape::order(inShape) == 'c')
                shape::shapeBuffer(shape.size(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

            return new ShapeList(newShape);
        }


        CUSTOM_OP_IMPL(crelu_bp, 2, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto epsilonNext = INPUT_VARIABLE(0);
            auto epsilon = OUTPUT_VARIABLE(0);

            // at first step we build fwd activation
            nd4j::ops::crelu<T> op;
            auto tmpResult = op.execute({input}, {}, {}); 
            auto actv = tmpResult->at(0);

            // now we do RELU backward pass
            auto lambda = LAMBDA_TT(_x, _e) {
                return _x > (T) 0.0f ? _e  : (T) 0.0f;
            };
            actv->applyPairwiseLambda(epsilon, lambda);

            // now we split updated array into 2 chunks along last dimension


            // and now we subtract two parts of epsilons and pass result out

            delete tmpResult;
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(crelu_bp) {
            auto inShape = inputShape->at(0);
            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }
    }
}