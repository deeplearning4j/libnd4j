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
    NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr
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
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
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
    
    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
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

        // if(isSameMode)                       // SAME        
        //     ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

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

    int oC = weightsPointShapeInfo ? weightsPointShapeInfo[indOC+1] : iC*mC;      // output channels (oC or iC*mC)

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


////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sconv2d_bp, 3, 2, false, 0, 9) {

    NDArray<T> *input        = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *gradO        = INPUT_VARIABLE(1);                                           // [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    NDArray<T> *weightsDepth = INPUT_VARIABLE(2);                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr 
    
    NDArray<T> *gradI  = OUTPUT_VARIABLE(0);                                                // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    NDArray<T> *gradWD = OUTPUT_VARIABLE(1);                                                // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW] (NCHW)
    NDArray<T> *gradWP = nullptr;                                                           // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *gradB  = nullptr;                                                           // [oC]

    if(block.width() == 4) {
        if((INPUT_VARIABLE(3))->rankOf() == 4) {
            weightsPoint = INPUT_VARIABLE(3);
            gradWP       = OUTPUT_VARIABLE(2);
        }
        else {
            bias  = INPUT_VARIABLE(3);
            gradB = OUTPUT_VARIABLE(2);
        }
    }
    else if(block.width() == 5) {
        weightsPoint = INPUT_VARIABLE(3);
        bias         = INPUT_VARIABLE(4);
        gradWP       = OUTPUT_VARIABLE(2);
        gradB        = OUTPUT_VARIABLE(3);
    }
        

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM SCONV2D_BP OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(gradO->rankOf()   == 4, 0, "CUSTOM SCONV2D_BP OP: rank of gradI (epsilon_next) array must be equal to 4 !");
    REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsDepth array must be equal to 4 !");
    if(weightsPoint) {
        REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsPoint array must be equal to 4 !");
        REQUIRE_TRUE(gradWP->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsPoint gradients array must be equal to 4 !");
    }
    if(bias) {
        REQUIRE_TRUE(bias->rankOf() == 1  || bias->rankOf()  == 2, 0, "CUSTOM SCONV2D_BP OP: rank of biases array must be equal to 1 or 2 !");           
        REQUIRE_TRUE(gradB->rankOf() == 1 || gradB->rankOf() == 2, 0, "CUSTOM SCONV2D_BP OP: rank of gradients biases array must be equal to 1 or 2 !");           
    }

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    REQUIRE_TRUE(weightsDepth->sizeAt(indWiC) == iC && weightsDepth->sizeAt(indWkH) == kH && weightsDepth->sizeAt(indWkH+1) == kW, 0, 
                "CUSTOM SCONV2D_BP OP: wrong shape of weightsDepth array! Should be [%i, %i, %i, %i], but got [%i, %i, %i, %i] instead !", 
                !isNCHW?kH:mC, !isNCHW?kW:iC, !isNCHW?iC:kH, !isNCHW?mC:kW, weightsDepth->sizeAt(0), weightsDepth->sizeAt(1), weightsDepth->sizeAt(2), weightsDepth->sizeAt(3));

    REQUIRE_TRUE(gradWD->sizeAt(indWiC) == iC && gradWD->sizeAt(indWkH) == kH && gradWD->sizeAt(indWkH+1) == kW, 0, 
                "CUSTOM SCONV2D_BP OP: wrong shape of gradients weightsDepth array! Should be [%i, %i, %i, %i], but got [%i, %i, %i, %i] instead !", 
                !isNCHW?kH:mC, !isNCHW?kW:iC, !isNCHW?iC:kH, !isNCHW?mC:kW, gradWD->sizeAt(0), gradWD->sizeAt(1), gradWD->sizeAt(2), gradWD->sizeAt(3));

    if(weightsPoint) {
        REQUIRE_TRUE(weightsPoint->sizeAt(indWmC) == oC && weightsPoint->sizeAt(indWiC) == iC*mC && weightsPoint->sizeAt(indWkH) == 1 && weightsPoint->sizeAt(indWkH+1) == 1, 0,
                    "CUSTOM SCONV2D_BP OP: wrong shape of weightsPoint array! Should be [%i, %i, %i, %i], but got [%i, %i, %i, %i] instead !", 
                    !isNCHW?1:oC, !isNCHW?1:iC*mC, !isNCHW?iC*mC:1, !isNCHW?oC:1, weightsPoint->sizeAt(0), weightsPoint->sizeAt(1), weightsPoint->sizeAt(2), weightsPoint->sizeAt(3));
        REQUIRE_TRUE(gradWP->sizeAt(indWmC) == oC && gradWP->sizeAt(indWiC) == iC*mC && gradWP->sizeAt(indWkH) == 1 && gradWP->sizeAt(indWkH+1) == 1, 0,
                    "CUSTOM SCONV2D_BP OP: wrong shape of gradients weightsPoint array! Should be [%i, %i, %i, %i], but got [%i, %i, %i, %i] instead !", 
                    !isNCHW?1:oC, !isNCHW?1:iC*mC, !isNCHW?iC*mC:1, !isNCHW?oC:1, gradWP->sizeAt(0), gradWP->sizeAt(1), gradWP->sizeAt(2), gradWP->sizeAt(3));         
    }
    if (bias) {
        REQUIRE_TRUE(oC == bias->lengthOf(),  0, "CUSTOM SCONV2D_BP OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
        REQUIRE_TRUE(oC == gradB->lengthOf(), 0, "CUSTOM SCONV2D_BP OP: length of gradients bias array must be equal to outChannels, but got %i instead", gradB->lengthOf());
    }
  
    // if (iC == 1) {
    //     nd4j_debug("CUSTOM SCONV2D_BP OP: for input_channels=1 this op is equivalent to standard conv2d_bp \n","");
    //     nd4j::ops::conv2d_bp<T> op;
    //     return op.execute(&block);
    // }
        
    // if weightsPoint != nullptr then perform point-wise backprop first, at the same time calculating gradWP    
    if (weightsPoint){           
        nd4j::ops::sconv2d<T> opFF;
        ResultSet<T>* resultFF = opFF.execute({input, weightsDepth}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});
        NDArray<T>* depthInput = resultFF->at(0);          // [bS, oH, oW, mC]  (NHWC) or [bS, mC, oH, oW] (NCHW)

        std::initializer_list<int> gradIDepthShape = !isNCHW ? std::initializer_list<int>({bS, oH, oW, iC*mC}) : std::initializer_list<int>({bS, iC*mC, oH, oW});
        NDArray<T>* gradIDepth = new NDArray<T>(gradIDepthShape, block.getWorkspace());             // [bS, oH, oW, iC*mC]  (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

        nd4j::ops::conv2d_bp<T> opBP;
        opBP.execute({depthInput, weightsPoint, bias, gradO}, {gradIDepth, gradWP, gradB}, {}, {1,1, 1,1, 0,0, 1,1, isSameMode, !isNCHW});      // in this case oH=iH and oW=iW
    
        gradO = gradIDepth;

        delete resultFF;
    }
    
    // -----prepare permutation arrays and axes for dot product ----- //
    std::vector<std::vector<int>> modifColumns = {{1,2,3,0,4,5}, {iC,kW*kH,bS*oH*oW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,kH,kW,bS,oH,oW] -> [iC,kW*kH,bS*oH*oW]
    std::vector<std::vector<int>> modifGradO1, modifGradO2, modifWeights;
    std::vector<int> reshapeArrForGradO;         
    
    if(!isNCHW) {                
        input = input->permute({0, 3, 1, 2});                                           // [bS, iH, iW, iC] -> [bS, iC, iH, iW] 
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                                
        reshapeArrForGradO = {bS, oH, oW, iC, mC};                                      // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,4,0,1,2},{iC, mC, bS*oH*oW}};                                 // [bS,oH,oW,iC,mC] -> [iC,mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        reshapeArrForGradO = {bS, iC, mC, oH, oW};                                      // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,2,0,3,4},{iC, mC, bS*oH*oW}};                                 // [bS,iC,mC,oH,oW] -> [iC,mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                              // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    

    // ----- calculation of gradWD ----- //
    NDArray<T>* gradOReshaped = gradO->reshape(gradO->ordering(), reshapeArrForGradO);
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradOReshaped, gradWD, modifColumns, modifGradO1, modifWeights);    // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] =  [iC, kH*kW, mC]
  
    // ----- calculate gradB if required  ----- //
    if(gradB && !weightsPoint) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,indOoH,indOoH+1});                           // sum over bS, oH, oW
        if(gradB != OUTPUT_VARIABLE(2)) 
            delete gradB;
    }

    //----- calculation of gradI -----//
    nd4j::NDArrayFactory<T>::tensorDot(weightsDepth, gradOReshaped, &columns, modifWeights, modifGradO2, modifColumns);  // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]    
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                                     // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]            
    
    delete gradOReshaped;

    if(!isNCHW) {
        delete input;
        delete gradI;
    }
    if(weightsPoint)
        delete gradO;
    
    return Status::OK();
    
}


DECLARE_SHAPE_FN(sconv2d_bp) {

    int* inputShapeInfo        = inputShape->at(0);
    int* weightsDepthShapeInfo = inputShape->at(2);
    int* weightsPointShapeInfo = nullptr;
    int* biasShapeInfo         = nullptr;

    if(block.width() == 4) {    
        if(inputShape->at(3)[0] == 4)
            weightsPointShapeInfo = inputShape->at(3);
        else 
            biasShapeInfo  = inputShape->at(3);
    }
    else if(block.width() == 5) {
        weightsPointShapeInfo = inputShape->at(3);
        biasShapeInfo         = inputShape->at(4);
    }

    int* gradIshapeInfo(nullptr), *gradWDshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsDepthShapeInfo, gradWDshapeInfo);

    int* gradWPshapeInfo(nullptr), *gradBshapeInfo(nullptr);
    
    if(weightsPointShapeInfo && biasShapeInfo) {        
        COPY_SHAPE(weightsPointShapeInfo, gradWPshapeInfo);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradWPshapeInfo, gradBshapeInfo);
    }

    if(weightsPointShapeInfo && !biasShapeInfo) {        
        COPY_SHAPE(weightsPointShapeInfo, gradWPshapeInfo);        
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradWPshapeInfo);
    }

    if(!weightsPointShapeInfo && biasShapeInfo) {        
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradBshapeInfo);
    }

    return SHAPELIST(gradIshapeInfo, gradWDshapeInfo);
}



}
}