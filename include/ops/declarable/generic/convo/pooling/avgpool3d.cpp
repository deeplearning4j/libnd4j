//
// created by Yurii Shyrma on 19.02.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew, 1, 1, false, 0, 10) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)
                                     
    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int paddingMode = INT_ARG(9);                                               // 0-SAME,  1-VALID
    int dataFormat  = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;     // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM AVGPOOL3D OP: rank of input array must be equal to 5 !");    
    
    if(!dataFormat) {
        input  = input->permute({0, 4, 1, 2, 3});                              // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        output = output->permute({0, 4, 1, 2, 3});                             // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW]
        
        // input->streamline('c');
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iD = input->sizeAt(2);           // input depth
    int iH = input->sizeAt(3);           // input height
    int iW = input->sizeAt(4);           // input width    
    int oD = output->sizeAt(2);          // output depth
    int oH = output->sizeAt(3);          // output height
    int oW = output->sizeAt(4);          // output width            

    REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM AVGPOOL3D OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width !");    
    REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM AVGPOOL3D OP: pad must not be greater than half of kernel size!");    
    
    if(!paddingMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 0, 0, 0);    

    int iStride = iC * iD * iH * iW;
    int oStride = iC * oD * oH * oW;
    
    for(int i = 0; i < bS; ++i)   
        ConvolutionUtils<T>::_avgPool3D(input->getBuffer() + i*iStride, output->getBuffer() + i*oStride, iC, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, true);        
   
    if(!dataFormat) {
        delete input;
        delete output;
    }
        
    return Status::OK();
}


DECLARE_SHAPE_FN(avgpool3dnew) {

    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int paddingMode = INT_ARG(9);                                               // 0-SAME,  1-VALID;
    int dataFormat  = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;     // 0-NDHWC, 1-NCDHW
    
    int indID = dataFormat == 0 ? 1 : 2;
    int indIC = dataFormat == 0 ? 4 : 1;
    int indOC = dataFormat == 0 ? 4 : 0;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                          // batch size
    int iD = inputShapeInfo[indID+1];                    // input depth
    int iH = inputShapeInfo[indID+2];                    // input height
    int iW = inputShapeInfo[indID+3];                    // input width
    int iC = inputShapeInfo[indIC+1];                    // input channels            

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, 0, 0, 0, iD, iH, iW, paddingMode);

    if(!paddingMode)                        // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 0, 0, 0);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    if (dataFormat) {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = iC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = iC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return new ShapeList(outputShapeInfo);
}



}
}