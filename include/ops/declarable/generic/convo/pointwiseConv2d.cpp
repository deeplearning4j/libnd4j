//
// Created by Yurii Shyrma 20.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(pointwise_conv2d, 2, 1, false, 0, 0) {

    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [1,  1,  iC, oC] (NHWC) or [oC, iC,  1,  1] (NCHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, iH, iW, oC] (NHWC) or [bS, oC, iH, iW] (NCHW)
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM POINTWISECONV2D OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM POINTWISECONV2D OP: rank of weights array must be equal to 4 !");    
    if(bias)
        REQUIRE_TRUE(bias->rankOf() == 1 || bias->rankOf() == 2, 0, "CUSTOM POINTWISECONV2D OP: rank of biases array must be equal to 1 or 2 !");           

    int kH = 1;                                                             // filter(kernel) height
    int kW = 1;                                                             // filter(kernel) width
    int sH = 1;                                                             // strides height
    int sW = 1;                                                             // strides width
    int pH = 0;                                                             // paddings height
    int pW = 0;                                                             // paddings width
    int dH = 1;                                                             // dilations height
    int dW = 1;                                                             // dilations width    
    int isNCHW = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;       // 1-NCHW, 0-NHWC
    
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    REQUIRE_TRUE(weights->sizeAt(indWiC) == iC && weights->sizeAt(indWkH) == kH && weights->sizeAt(indWkH+1) == kW, 0, "CUSTOM POINTWISECONV2D OP: wrong shape of weights array !");
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() <= 2,    0, "CUSTOM POINTWISECONV2D OP: rank of biases array must be equal to 1 or 2!");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM POINTWISECONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            
    
    nd4j::ops::conv2d<T> op;
    Nd4jStatus status = op.execute({input, weights, bias}, {output}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, 1/*isSameMode*/, !isNCHW});

    if (status != Status::OK())
        return status;
    
    return Status::OK();
}


DECLARE_SHAPE_FN(pointwise_conv2d) {
    
    int* inputShapeInfo  = inputShape->at(0);        
    int* outputShapeInfo = nullptr;
    COPY_SHAPE(inputShapeInfo, outputShapeInfo);  

    // do not forget to put oC instead of iC in outputShapeInfo
    int isNCHW = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;   // 0-NCHW,  1-NHWC
    int indOC  = isNCHW == 1 ? 0 : 3;
    int indIC  = isNCHW == 1 ? 1 : 3;
    int oC     =  inputShape->at(1)[indOC+1];                           // output channels
    outputShapeInfo[indIC + 1] = oC;                                   

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);        
}

}
}