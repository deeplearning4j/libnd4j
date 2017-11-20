//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        /**
         * 
         * Kernel
         * Stride
         * Padding
         */
        CUSTOM_OP_IMPL(conv1d, 2, 1, false, 0, 3) {
            auto input = INPUT_VARIABLE(0);
            auto weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            if (block.width() > 2)
                bias = INPUT_VARIABLE(2);

            int kernel = INT_ARG(0);
            int stride = INT_ARG(1);
            int padding = INT_ARG(2);

            REQUIRE_TRUE(weights->rankOf() == 3, 0, "Conv1D requires 3D input, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(input->rankOf() == 3, 0, "Conv1D requires 3D input, but got %iD instead", input->rankOf());
            
            auto z = OUTPUT_VARIABLE(0);

            auto _input = input->reshape(input->ordering(),{input->sizeAt(0), input->sizeAt(1), input->sizeAt(2), 1});
            auto _weights = weights->reshape(weights->ordering(),{weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2), 1});

            nd4j::ops::conv2d<T> op;
            auto result = op.execute({_input, _weights, bias}, {}, {kernel, 1, stride, 1, padding, 0, 1, 1, 0});
            auto tmpZ = result->at(0);

            tmpZ->reshapei(tmpZ->ordering(), {tmpZ->sizeAt(0), tmpZ->sizeAt(1), tmpZ->sizeAt(2)});

            z->assign(tmpZ);

            delete result;
            delete _input;
            delete _weights;
            
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(conv1d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            const int kY = INT_ARG(0);
            const int sY = INT_ARG(1);
            int pY = INT_ARG(2);
            const bool isSameMode = false;

            int oY = 0;
            int oX = 0;

            const int batchSize = inShape[1];
            const int outDepth = wShape[1];
            const int inY = inShape[3];
            const int inX = 1; // constant value

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, 1, sY, 1, pY, 0, 1, 1, inY, inX, isSameMode);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(3), int);
            std::vector<int> shape({batchSize, outDepth, oY});
            shape::shapeBuffer(3, shape.data(), newShape);

            return new ShapeList(newShape);
        }

        CUSTOM_OP_IMPL(conv1d_bp, 3, 2, false, 0, 3) {

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(conv1d_bp) {

            return new ShapeList(nullptr);
        }
    }
}