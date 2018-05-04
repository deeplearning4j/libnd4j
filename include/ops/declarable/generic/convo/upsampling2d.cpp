//
// @author raver119, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 03.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_upsampling2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling2d, 1, 1, false, 0, 2) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);             // [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC) 
    NDArray<T>* output = OUTPUT_VARIABLE(0);            // [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
            
    const int factorH = INT_ARG(0);
    const int factorW = INT_ARG(1);
    const int isNCHW  = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;       // 1-NCHW,  0-NHWC

    REQUIRE_TRUE(input->rankOf() == 4, 0, "UPSAMPLING2D op: input should be 4D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(output->rankOf() == 4, 0, "UPSAMPLING2D op: output should be 4D, but got %i instead!", output->rankOf());

    ConvolutionUtils<T>::upsampling2d(*input, *output, factorH, factorW, (bool)isNCHW);

    return Status::OK();
}
DECLARE_SYN(upsampling, upsampling2d);


        
DECLARE_SHAPE_FN(upsampling2d) {
    
    int* inputShapeInfo = inputShape->at(0);
    
    REQUIRE_TRUE(inputShapeInfo[0] == 4, 0, "UPSAMPLING2D op: input should be 4D, but got %i instead!", inputShapeInfo[0]);

    const int factorH = INT_ARG(0);
    const int factorW = INT_ARG(1);
    const int isNCHW  = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;       // 1-NCHW,  0-NHWC

    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo[0]), int);

    outputShapeInfo[0] = inputShapeInfo[0];
    outputShapeInfo[1] = inputShapeInfo[1];

    if(isNCHW) {
        outputShapeInfo[2] = inputShapeInfo[2];
        outputShapeInfo[3] = inputShapeInfo[3] * factorH;
        outputShapeInfo[4] = inputShapeInfo[4] * factorW;
    }
    else {        
        outputShapeInfo[2] = inputShapeInfo[2] * factorH;
        outputShapeInfo[3] = inputShapeInfo[3] * factorW;
        outputShapeInfo[4] = inputShapeInfo[4];
    }

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


        //////////////////////////////////////////////////////////////////////////
        /**
         * Upsampling backprop implementation, based on pytorch
         *
         * Input[0] - preoutput result
         * Input[1] - gradients from next node/layer
         *
         * Output[0] - gradient for this node
         *
         * IArgs map:
         * IArgs[0] - scale factor
         */
        CUSTOM_OP_IMPL(upsampling2d_bp, 2, 1, false, 0, 1) {
            //NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* gradientNext = INPUT_VARIABLE(1);
            NDArray<T>* output = this->getZ(block);
            int scale_factor = INT_ARG(0);


            int dW = scale_factor;
            int dH = scale_factor;
            int xDim = output->rankOf() - 2;
            int yDim = output->rankOf() - 1;

            // dims
            int idim = output->rankOf();  // Guaranteed to be between 3 and 5
            int isz0 = output->sizeAt(0);
            int isz1 = output->sizeAt(1);
            int isz2 = output->sizeAt(2);
            int isz3 = 1;
            if (idim > 3) {
                isz3 = output->sizeAt(3);
            }

            output->assign(0.0);

            // perform the upsampling
            int i0, i1, i2, i3, isrc, idst, x, y;
            int iin[4];  // Input indices
            int iout[4];  // Output indices

            for (i0 = 0; i0 < isz0; i0++) {
                iin[0] = i0;
                iout[0] = i0;
                for (i1 = 0; i1 < isz1; i1++) {
                    iin[1] = i1;
                    iout[1] = i1;
                    for (i2 = 0; i2 < isz2; i2++) {
                        iin[2] = i2;
                        iout[2] = i2;
                        for (i3 = 0; i3 < isz3; i3++) {
                            iin[3] = i3;
                            iout[3] = i3;

                            idst = i0 * output->stridesOf()[0] + i1 * output->stridesOf()[1] + i2 * output->stridesOf()[2];
                            if (idim > 3) {
                                idst += i3 * output->stridesOf()[3];
                            }

                            // Now accumulate the gradients from gradOutput
                            for (y = 0; y < dH; y++) {
                                for (x = 0; x < dW; x++) {
                                    iout[xDim] = dW * iin[xDim] + x;
                                    iout[yDim] = dH * iin[yDim] + y;
                                    isrc = iout[0] * gradientNext->stridesOf()[0] + iout[1] * gradientNext->stridesOf()[1] + iout[2] * gradientNext->stridesOf()[2];
                                    if (idim > 3) {
                                        isrc += iout[3] * gradientNext->stridesOf()[3];
                                    }
                                    output->getBuffer()[idst] += gradientNext->getBuffer()[isrc];
                                }
                            }
                        }
                    }
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(upsampling_bp, upsampling2d_bp);
        DECLARE_SHAPE_FN(upsampling2d_bp) {
            auto inShape = inputShape->at(0);

            int scale = INT_ARG(0);

            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            int shape[] = {shape::shapeOf(inShape)[0], shape::shapeOf(inShape)[1], shape::shapeOf(inShape)[2] / scale, shape::shapeOf(inShape)[3] / scale};
            shape::shapeBuffer(4, shape, newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif