//
// @authors raver119@gmail.com and Yurii Shyrma
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
  
        CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
            
            NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, oC, iC] (NHWC) or [iC, oC, kH, kW] (NCHW)
            NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
            NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    
            REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D OP: rank of input array must be equal to 4 !");
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D OP: rank of weights array must be equal to 4 !");
                                     
            int kH = INT_ARG(0);                                                        // filter(kernel) height
            int kW = INT_ARG(1);                                                        // filter(kernel) width
            int sH = INT_ARG(2);                                                        // strides height
            int sW = INT_ARG(3);                                                        // strides width
            int pH = INT_ARG(4);                                                        // paddings height
            int pW = INT_ARG(5);                                                        // paddings width
            int dH = INT_ARG(6);                                                        // dilations height
            int dW = INT_ARG(7);                                                        // dilations width
            int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
            int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *weights, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
            oC = output->sizeAt(indIOioC);                              // getSizesAndIndexesConv2d gives wrong value of oC in deconv case

            REQUIRE_TRUE(weights->sizeAt(indWoC) == oC && weights->sizeAt(indWkH) == kH && weights->sizeAt(indWkH+1) == kW, 0, "CUSTOM DECONV2D OP: wrong shape of weights array !");    
            if (bias)
                REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D OP: wrong shape of array with biases !");

            std::vector<int> permutForColumns;

            if(!isNCHW) {
                output  = output->permute({0, 3, 1, 2});                                // [bS, oH, oW, oC] -> [bS, oC, oH, oW] 
                permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [kH, kW, oC, bS, iH, iW]
            }
            else 
                permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [oC, kH, kW, bS, iH, iW]
            
            if(isSameMode)                       // SAME        
                ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);                
            
            NDArray<T> columns(input->ordering(), {bS, oC, kH, kW, iH, iW}, block.getWorkspace());        
            std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) oH, (T) oW, (T) dH, (T) dW});          

            //----- calculation of output -----//
            // NHWC: [kH, kW, oC, iC] x [bS, iH, iW, iC] = [kH, kW, oC, bS, iH, iW]
            // NCHW: [iC, oC, kH, kW] x [bS, iC, iH, iW] = [oC, kH, kW, bS, iH, iW]
            nd4j::NDArrayFactory<T>::tensorDot(weights, input, &columns, {indWiC}, {indIOioC}, permutForColumns);  
            columns.template applyTransform<simdOps::Col2Im<T>>(output, extrasCol2Im.data());                            // [bS, oC, kH, kW, iH, iW] is de-convoluted to [bS, oC, oH, oW]
           
            //----- add biases if required -----//
            if(bias)
                output->template applyBroadcast<simdOps::Add<T>>({1}, bias);

            if(!isNCHW) 
                delete output;    
    
            return Status::OK();

        }

        DECLARE_SHAPE_FN(deconv2d) {

            int kH = INT_ARG(0);                                                        // filter(kernel) height
            int kW = INT_ARG(1);                                                        // filter(kernel) width
            int sH = INT_ARG(2);                                                        // strides height
            int sW = INT_ARG(3);                                                        // strides width
            int pH = INT_ARG(4);                                                        // paddings height
            int pW = INT_ARG(5);                                                        // paddings width
            int dH = INT_ARG(6);                                                        // dilations height
            int dW = INT_ARG(7);                                                        // dilations width
            int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
            int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NDHWC, 1-NCDHW

            int indIH = isNCHW == 1 ? 2 : 1;
            int indIC = isNCHW == 1 ? 1 : 3;
            int indOC = isNCHW == 1 ? 1 : 2;

            int* inputShapeInfo   = inputShape->at(0);
            int* weightsShapeInfo = inputShape->at(1);

            int bS = inputShapeInfo[1];                         // batch size
            int iH = inputShapeInfo[indIH+1];                   // input height
            int iW = inputShapeInfo[indIH+2];                   // input width
            int iC = inputShapeInfo[indIC+1];                   // input channels        
            int oC = weightsShapeInfo[indOC+1];                 // output channels

            int oH, oW;                                         // output height, width
            ConvolutionUtils<T>::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
            int* outputShapeInfo = nullptr;
            ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

            outputShapeInfo[0] = 4;
            outputShapeInfo[1] = bS;

            if (isNCHW) {
                outputShapeInfo[2] = oC;
                outputShapeInfo[3] = oH;
                outputShapeInfo[4] = oW;
            } else {
                outputShapeInfo[2] = oH;
                outputShapeInfo[3] = oW;
                outputShapeInfo[4] = oC;
            }
    
            shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

            return SHAPELIST(outputShapeInfo);
        }


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(deconv2d_bp, 3, 2, false, 0, 9) {
            
            NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
            NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kH, kW, oC, iC] (NDHWC) or [iC, oC, kH, kW] (NCDHW)
            NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
            NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

            NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), gradI
            NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, oC, iC] (NDHWC) or [iC, oC, kH, kW] (NCDHW)
            NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

            REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D_BP OP: rank of input array must be equal to 4 !");
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D_BP OP: rank of weights array must be equal to 4 !");
            REQUIRE_TRUE(gradO->rankOf()   == 4, 0, "CUSTOM DECONV2D_BP OP: rank of gradO array must be equal to 4 !");

            int kH = INT_ARG(0);                                                        // filter(kernel) height
            int kW = INT_ARG(1);                                                        // filter(kernel) width
            int sH = INT_ARG(2);                                                        // strides height
            int sW = INT_ARG(3);                                                        // strides width
            int pH = INT_ARG(4);                                                        // paddings height
            int pW = INT_ARG(5);                                                        // paddings width
            int dH = INT_ARG(6);                                                        // dilations height
            int dW = INT_ARG(7);                                                        // dilations width
            int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
            int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *weights, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
            oC = output->sizeAt(indIOioC);                              // getSizesAndIndexesConv2d gives wrong value of oC in deconv case

            int trueoH, trueoW;          // true output height, width
            ConvolutionUtils<T>::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

            REQUIRE_TRUE(gradO->sizeAt(0)==bS   && gradO->sizeAt(indOoH)==trueoH && gradO->sizeAt(indOoH+1)==trueoW && gradO->sizeAt(indIOioC)==oC, 0,  "CUSTOM CONV2D_BP OP: wrong shape of gradient_output (next epsilon) array !");
            REQUIRE_TRUE(weights->sizeAt(indWoC)==oC && weights->sizeAt(indWiC)==iC && weights->sizeAt(indWkH)==kH && weights->sizeAt(indWkH+1)==kW, 0, "CUSTOM CONV2D_BP OP: wrong shape of weights array !");
            if(bias)
                REQUIRE_TRUE(bias->rankOf()<=2 && bias->lengthOf()==oC, 0, "CUSTOM CONV2D_BP OP: wrong shape of biases array !");

            // pass gradI through conv2d_ff 
            nd4j::ops::conv2d<T> conv2d;
            ResultSet<T>* results = conv2d.execute({gradO, weights}, {gradI}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW,  paddingMode,  isNCHW});
            if (results->status() != ND4J_STATUS_OK)
                return results->status();



            nd4j::ops::conv2d<T> op;
            Nd4jStatus r1 = op.execute({gradO, weights}, {gradI}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, isSameMode, isNCHW});
            if (r1 != ND4J_STATUS_OK)
                return r1;

            int oY = 0;
            int oX = 0;
            int inY = gradO->sizeAt(2);
            int inX = gradO->sizeAt(3);

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kH, kW, sH, sW, pH, pW, dH, dW, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pH, pW, oY, oX, inY, inX, kH, kW, sH, sW, dH, dW);
            }

            std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, isSameMode ? (T) 1.0f : (T) 0.0f, 0.0});
            auto columns = new NDArray<T>('c', {input->sizeAt(0), weights->sizeAt(1), kH, kW, oY, oX });
            gradO->template applyTransform<simdOps::Im2col<T>>(columns, extrasIm2Col.data());

            auto gW = NDArrayFactory<T>::tensorDot(input, columns, {0, 2, 3}, {0, 4, 5});
            gradW->assign(gW);

            delete gW;
            delete columns;

            if (gradB != nullptr) {
                auto sum = gradO->template reduceAlongDimension<simdOps::Sum<T>>({0, 2, 3});
                gradB->assign(sum);
                delete sum;

                STORE_3_RESULTS(*gradI, *gradW, *gradB);
            } else {
                STORE_2_RESULTS(*gradI, *gradW);
            }



            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(deconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);
            int* eShape = nullptr;
            int* bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 4) {
                bShape = inputShape->at(2);
                eShape = inputShape->at(3);
            } else {
                eShape = inputShape->at(2);
            }

            int *newInShape;
            int *newWShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapes = SHAPELIST(newInShape, newWShape);

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapes->push_back(newBShape);
            }

            return shapes;
        }

    }
}