//
// Created by raver119 on 29/10/17.
// changed by Yurii Shyrma 20.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <memory>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(sconv2d, 2, 1, false, 0, 9) {

    NDArray<T> *input        = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *weightsDepth = INPUT_VARIABLE(1);                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, pC] (NHWC) or [pC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr else oC = pC
    NDArray<T> *output       = OUTPUT_VARIABLE(0);                                          // [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)
    
    if(block.width() == 3) {
        if((INPUT_VARIABLE(2))->rankOf() == 4)
            weightsPoint = INPUT_VARIABLE(2);
        else
            bias = INPUT_VARIABLE(2);
    }
    else if(block.width() == 4) {
        weightsPoint = INPUT_VARIABLE(2);
        bias = INPUT_VARIABLE(3);
    }

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM SCONV2D OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsDepth array must be equal to 4 !");
    if(weightsPoint)
        REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsPoint array must be equal to 4 !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf() == 1 || bias->rankOf() == 2, 0, "CUSTOM SCONV2D OP: rank of biases array must be equal to 1 or 2 !");           

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
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    REQUIRE_TRUE(weightsDepth->sizeAt(indWiC) == iC && weightsDepth->sizeAt(indWkH) == kH && weightsDepth->sizeAt(indWkH+1) == kW, 0, "CUSTOM SCONV2D OP: wrong shape of weightsDepth array !");
    if(weightsPoint)
        REQUIRE_TRUE(weightsPoint->sizeAt(indWmC) == oC && weightsPoint->sizeAt(indWiC) == iC*mC && weightsPoint->sizeAt(indWkH) == 1 && weightsPoint->sizeAt(indWkH+1) == 1, 0, "CUSTOM SCONV2D OP: wrong shape of weightsPoint array !");    
    if (bias)        
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM SCONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
  
    if (iC == 1) {
        nd4j_debug("CUSTOM SCONV2D OP: for input_channels=1 this op is equivalent to standard conv2d\n","");
        nd4j::ops::conv2d<T> c2d;
        return c2d.execute(&block);
    }
    
    Nd4jStatus status;

    // ----- perform depthwise convolution (oC = iC*mC) ----- //
    if (!weightsPoint) {        // weightsPoint == nullptr
        
        nd4j::ops::depthwise_conv2d<T> op;
        status = op.execute({input, weightsDepth, bias}, {output}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});                                   
    }
    
    // ----- perform pointwise convolution (oC = pC) ----- //
    else {             

        std::vector<std::vector<int>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
        std::vector<std::vector<int>> modifWeights;
        std::vector<int> reshapeForinputConv2d, permutForInputConv2d;

        if(!isNCHW) {                
            input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
            permutForInputConv2d  = {1, 2, 0, 3};                                           // [iC, bS, oH*oW, mC] -> [bS, oH*oW, iC, mC]
            reshapeForinputConv2d = {bS, oH, oW, iC*mC};
            modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
        }
        else {
            permutForInputConv2d  = {1, 0, 3, 2};                                           // [iC, bS, oH*oW, mC] -> [bS, iC, mC, oH*oW]
            reshapeForinputConv2d = {bS, iC*mC, oH, oW};        
            modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};                                       // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
        }

        if(isSameMode)                       // SAME        
            ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

        NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());
        std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
        input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                            // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
                    
        NDArray<T>* inputConv2d = NDArrayFactory<T>::tensorDot(&columns, weightsDepth, modifColumns, modifWeights);   // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
        inputConv2d->reshapei(input->ordering(), {iC, bS, oH*oW, mC});      // [iC, bS*oH*oW, mC] -> [iC, bS, oH*oW, mC]
        inputConv2d->permutei(permutForInputConv2d);                        // [iC, bS, oH*oW, mC] -> [bS, oH*oW, iC, mC] / [bS, iC, mC, oH*oW]
        inputConv2d->reshapei(reshapeForinputConv2d);                       // [bS, oH*oW, iC, mC] / [bS, iC, mC, oH*oW]  -> [bS, oH, oW, iC*mC] / [bS, iC*mC, oH, oW]
        
        // perform usual conv2d using inputConv2d as input and weightsPoint as weights
        nd4j::ops::conv2d<T> op;
        status = op.execute({inputConv2d, weightsPoint, bias}, {output}, {}, {1,1, 1,1, 0,0, 1,1, isSameMode, !isNCHW});

        delete inputConv2d;
        if(!isNCHW)
            delete input;        
    }
  
    if (status != ND4J_STATUS_OK)
        return status;
    
    
    return Status::OK();
}


DECLARE_SHAPE_FN(sconv2d) {
    
    int* inputShapeInfo        = inputShape->at(0);
    int* weightsDepthShapeInfo = inputShape->at(1);
    int* weightsPointShapeInfo = nullptr;

    if(block.width() == 3 && inputShape->at(2)[0] == 4)        
        weightsPointShapeInfo = inputShape->at(2);
    else if(block.width() == 4) 
        weightsPointShapeInfo = inputShape->at(2);        
    
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
    int indOC = isNCHW == 1 ? 0 : 3;
    
    int bS = inputShapeInfo[1];                         // batch size
    int iH = inputShapeInfo[indIH+1];                   // input height
    int iW = inputShapeInfo[indIH+2];                   // input width
    int iC = inputShapeInfo[indIC+1];                   // input channels        
    int mC = weightsDepthShapeInfo[indOC+1];            // channel multiplier

    int oC = weightsPointShapeInfo ? weightsPointShapeInfo[indOC+1] : iC*mC;      // output channels (pC or iC*mC)

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
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



// CUSTOM_OP_IMPL(sconv2d_bp, 3, 2, false, 0, 9) {

//     NDArray<T> *input        = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
//     NDArray<T> *gradO        = INPUT_VARIABLE(1);                                           // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
//     NDArray<T> *weightsDepth = INPUT_VARIABLE(2);                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
//     NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, pC] (NHWC) or [pC, iC*mC, 1, 1] (NCHW)
//     NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr else oC = pC    
    
//     NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
//     NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
//     NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

//     if(block.width() == 3) {
//         if((INPUT_VARIABLE(2))->rankOf() == 4)
//             weightsPoint = INPUT_VARIABLE(2);
//         else
//             bias = INPUT_VARIABLE(2);
//     }
//     else if(block.width() == 4) {
//         weightsPoint = INPUT_VARIABLE(2);
//         bias = INPUT_VARIABLE(3);
//     }

//     REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM SCONV2D OP: rank of input array must be equal to 4 !");
//     REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsDepth array must be equal to 4 !");
//     if(weightsPoint)
//         REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsPoint array must be equal to 4 !");
//     if(bias)
//         REQUIRE_TRUE(bias->rankOf() == 1 || bias->rankOf() == 2, 0, "CUSTOM SCONV2D OP: rank of biases array must be equal to 1 or 2 !");           

//     int kH = INT_ARG(0);                                                        // filter(kernel) height
//     int kW = INT_ARG(1);                                                        // filter(kernel) width
//     int sH = INT_ARG(2);                                                        // strides height
//     int sW = INT_ARG(3);                                                        // strides width
//     int pH = INT_ARG(4);                                                        // paddings height
//     int pW = INT_ARG(5);                                                        // paddings width
//     int dH = INT_ARG(6);                                                        // dilations height
//     int dW = INT_ARG(7);                                                        // dilations width
//     int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
//     int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC
    
//     int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
//     int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
//     ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
//     mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

//     REQUIRE_TRUE(weightsDepth->sizeAt(indWiC) == iC && weightsDepth->sizeAt(indWkH) == kH && weightsDepth->sizeAt(indWkH+1) == kW, 0, "CUSTOM SCONV2D OP: wrong shape of weightsDepth array !");
//     if(weightsPoint)
//         REQUIRE_TRUE(weightsPoint->sizeAt(indWmC) == oC && weightsPoint->sizeAt(indWiC) == iC*mC && weightsPoint->sizeAt(indWkH) == 1 && weightsPoint->sizeAt(indWkH+1) == 1, 0, "CUSTOM SCONV2D OP: wrong shape of weightsPoint array !");    
//     if (bias)        
//         REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM SCONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
  
//     if (iC == 1) {
//         nd4j_debug("CUSTOM SCONV2D OP: for input_channels=1 this op is equivalent to standard conv2d\n","");
//         nd4j::ops::conv2d<T> c2d;
//         return c2d.execute(&block);
//     }
    
//     Nd4jStatus status;

//     // ----- perform depthwise convolution (oC = iC*mC) ----- //
//     if (!weightsPoint) {        // weightsPoint == nullptr
        
//         nd4j::ops::depthwise_conv2d<T> op;
//         status = op.execute({input, weightsDepth, bias}, {output}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});                                   
//     }
    
//     // ----- perform pointwise convolution (oC = pC) ----- //
//     else {             

//         std::vector<std::vector<int>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
//         std::vector<std::vector<int>> modifWeights;
//         std::vector<int> reshapeForinputConv2d, permutForInputConv2d;

//         if(!isNCHW) {                
//             input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
//             permutForInputConv2d  = {1, 2, 0, 3};                                           // [iC, bS, oH*oW, mC] -> [bS, oH*oW, iC, mC]
//             reshapeForinputConv2d = {bS, oH, oW, iC*mC};
//             modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
//         }
//         else {
//             permutForInputConv2d  = {1, 0, 3, 2};                                           // [iC, bS, oH*oW, mC] -> [bS, iC, mC, oH*oW]
//             reshapeForinputConv2d = {bS, iC*mC, oH, oW};        
//             modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};                                       // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
//         }

//         if(isSameMode)                       // SAME        
//             ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

//         NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());
//         std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
//         input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                            // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
                    
//         NDArray<T>* inputConv2d = NDArrayFactory<T>::tensorDot(&columns, weightsDepth, modifColumns, modifWeights);   // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
//         inputConv2d->reshapei(input->ordering(), {iC, bS, oH*oW, mC});      // [iC, bS*oH*oW, mC] -> [iC, bS, oH*oW, mC]
//         inputConv2d->permutei(permutForInputConv2d);                        // [iC, bS, oH*oW, mC] -> [bS, oH*oW, iC, mC] / [bS, iC, mC, oH*oW]
//         inputConv2d->reshapei(reshapeForinputConv2d);                       // [bS, oH*oW, iC, mC] / [bS, iC, mC, oH*oW]  -> [bS, oH, oW, iC*mC] / [bS, iC*mC, oH, oW]
        
//         // perform usual conv2d using inputConv2d as input and weightsPoint as weights
//         nd4j::ops::conv2d<T> op;
//         status = op.execute({inputConv2d, weightsPoint, bias}, {output}, {}, {1,1, 1,1, 0,0, 1,1, isSameMode, !isNCHW});

//         delete inputConv2d;
//         if(!isNCHW)
//             delete input;        
//     }
  
//     if (status != ND4J_STATUS_OK)
//         return status;
    
    
//     return Status::OK();
// }



        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(sconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *epsilonNext = INPUT_VARIABLE(1);
            NDArray<T> *weightsDepth = INPUT_VARIABLE(2);
            NDArray<T> *weightsPoint = nullptr;
            NDArray<T> *bias = nullptr;

            // bias is still optional
            if (block.width() == 4) {
                auto tmp = INPUT_VARIABLE(3);
                if (tmp->rankOf() == 4)
                    weightsPoint = tmp;
                else
                    bias = tmp;
            } else if (block.width() == 5) {
                weightsPoint = INPUT_VARIABLE(3);
                bias = INPUT_VARIABLE(4);
            }

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead",
                         weightsDepth->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead",
                         epsilonNext->rankOf());

            if (weightsPoint != nullptr) {
                REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0,
                             "Weights for point-wise convolution should be 4d, but got %D instead",
                             weightsPoint->rankOf());
                REQUIRE_TRUE(weightsPoint->sizeAt(2) == 1 && weightsPoint->sizeAt(3) == 1, 1,
                             "Point-wise weights should be [1, 1], but got [%i, %i] instead", weightsPoint->sizeAt(2),
                             weightsPoint->sizeAt(3));
            }

            NDArray<T> *epsilon = OUTPUT_VARIABLE(0);
            NDArray<T> *gradWD = OUTPUT_VARIABLE(1);
            NDArray<T> *gradWP = nullptr;
            NDArray<T> *gradB = nullptr;

            if (weightsPoint != nullptr)
                gradWP = OUTPUT_VARIABLE(2);

            if (bias != nullptr)
                gradB = OUTPUT_VARIABLE(3);


            // now we're just launching depth-wise bp step

            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            int oY = epsilonNext->sizeAt(2);
            int oX = epsilonNext->sizeAt(3);

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weightsDepth->shapeOf()[0];
            const int inDepth = weightsDepth->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            // if weightsPoint are defiend - then we're going to do point-wise backprop first
            NDArray<T> *epsilon_;
            if (weightsPoint != nullptr) {
                nd4j::ops::sconv2d<T> opFF;
                auto result = opFF.execute({input, weightsDepth}, {}, {kY, kX, sY, sX, pY, pX, dY, dX, isSameMode ? 1 : 0});
                auto depthInput = result->at(0);

                nd4j::ops::conv2d_bp<T> opBP;

                epsilon_ = new NDArray<T>('c', {batchSize, weightsDepth->sizeAt(0) * weightsDepth->sizeAt(1), oY, oX});

                if (bias == nullptr)
                    opBP.execute({depthInput, weightsPoint, epsilonNext}, {epsilon_, gradWP}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0});
                else
                    opBP.execute({depthInput, weightsPoint, bias, epsilonNext}, {epsilon_, gradWP, gradB}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0});

                epsilonNext = epsilon_;

                delete result;
            }


            bool hasCol = CHECK_STASH("im2col");
            NDArray<T> *col = nullptr;
            if (hasCol)
                col = UNSTASH("im2col")
            else {
                col = new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX});

                // col2d now has shape of [bS, inDepth, kY, kX, oY, oX]
                std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX,
                                             isSameMode ? (T) 1.0f : (T) 0.0f, 0.0});

                input->template applyTransform<simdOps::Im2col<T>>(col, extrasIm2Col.data());
            }

//            epsilonNext->printShapeInfo("eps next");

            /*
             gy_ = gy.reshape((B, C, D, IY * IX)).transpose(1, 2, 0, 3).reshape((C, D, B * IY * IX))
             */
            auto eN_ = epsilonNext->reshape('c', {batchSize, inDepth, outDepth, oY * oX});
            eN_->permutei({1, 2, 0, 3});
            eN_->reshapei('c', {inDepth, outDepth, batchSize * oY * oX});

            auto col_ = col->permute({1, 0, 4, 5, 2, 3});
            col_->reshapei('c', {inDepth, batchSize * oY * oX, kY * kX});

            /*
             # (C, D, B*IY*IX), (C, B*IY*IX, KY*KX) -> (C, D, KY*KX)
                gW_ = _matmul(gy_, c_, xp)
             */

            // calculating wieghts gradients here
            //auto gW_ = gradW->reshape('c', {inDepth, outDepth, kY * kX});
            auto gW_ = NDArrayFactory<T>::mmulHelper(eN_, col_);

            gW_->reshapei('c', {inDepth, outDepth, kY, kX});
            gW_->permutei({1, 0, 2, 3});
            gradWD->assign(gW_);

            delete gW_;
            delete col_;
            if (!hasCol)
                delete col;

            // calculating epsilon here
            auto w_ = weightsDepth->permute({1, 2, 3, 0});
            w_->reshapei('c', {inDepth, kY * kX, outDepth});

            auto gcol = NDArrayFactory<T>::mmulHelper(w_, eN_);
            gcol->reshapei('c', {inDepth, kY, kX, batchSize, oY, oX});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::vector<T> extrasCol2Im({(T) sY, (T) sX, (T) pY, (T) pX, (T) inY, (T) inX, (T) dY, (T) dX,
                                         isSameMode ? (T) 1.0f : (T) 0.0f});

            // we're sure that col2im result will have the same size as original image
            //auto rCol = new NDArray<T>('c', {batchSize, inDepth, inY, inX});
            gcol->template applyTransform<simdOps::Col2Im<T>>(epsilon, extrasCol2Im.data());


            delete eN_;
            delete gcol;
            delete w_;


            if (weightsPoint == nullptr) {
                if (bias != nullptr) {
                    // calculating gradB, if defined
                    auto eN_ = epsilonNext->permute({0, 2, 3, 1});
                    auto sum = eN_->template reduceAlongDimension<simdOps::Sum<T>>({0, 1, 2});
                    gradB->assign(sum);
                    delete sum;

                    STORE_3_RESULTS(*epsilon, *gradWD, *gradB);
                } else {
                    STORE_2_RESULTS(*epsilon, *gradWD);
                }
            } else {
                if (bias != nullptr) {
                    STORE_4_RESULTS(*epsilon, *gradWD, *gradWP, *gradB);
                } else {
                    STORE_3_RESULTS(*epsilon, *gradWD, *gradWP);
                }
            }

            if (weightsPoint != nullptr)
                delete epsilonNext;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(sconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto eShape = inputShape->at(1);
            auto wdShape = inputShape->at(2);
            int *wpShape = nullptr;
            int *bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 5) {
                wpShape = inputShape->at(3);
                bShape = inputShape->at(4);
            } else if (inputShape->size() == 4) {
                auto tmp = inputShape->at(3);
                if (shape::rank(tmp) == 4)
                    wpShape = tmp;
                else
                    bShape = tmp;
            }

            int *newInShape;
            int *newWdShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWdShape, block.getWorkspace(), shape::shapeInfoLength(wdShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWdShape, wdShape, shape::shapeInfoByteLength(wdShape));

            auto shapes = SHAPELIST(newInShape, newWdShape);

            // FIXME: remove memcpy here
            if (wpShape != nullptr) {
                int *newWpShape;
                ALLOCATE(newWpShape, block.getWorkspace(), shape::shapeInfoLength(wpShape), int);
                memcpy(newWpShape, wpShape, shape::shapeInfoByteLength(wpShape));

                shapes->push_back(newWpShape);
            }

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