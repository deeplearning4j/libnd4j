//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.03.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew, 1, 1, false, 0, 13) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)
                                     
    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode  = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int extraParam0 = INT_ARG(13);
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedOutputShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    REQUIRE_TRUE(expectedOutputShape == ShapeUtils<T>::shapeAsString(output), 0, "AVGPOOL3D op: wrong shape of output array, expected is %s, but got %s instead !", expectedOutputShape.c_str(), ShapeUtils<T>::shapeAsString(output).c_str());

    if(!isNCDHW) {
        input  = input->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        output = output->permute({0, 4, 1, 2, 3});                                                      // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
    }    

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    
    
    T extraParams[] = {(T)kD, (T)kH, (T)kW, (T)sD, (T)sH, (T)sW, (T)pD, (T)pH, (T)pW, (T)dD, (T)dH, (T)dW, (T)extraParam0, (T)!isSameMode};
    ConvolutionUtils<T>::pooling3d(*input, *output, extraParams);
   
    if(!isNCDHW) {              
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
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 0-NDHWC, 1-NCDHW    
    
    int* inputShapeInfo = inputShape->at(0);

    int idxID, idxIC;    
    if(isNCDHW) { idxID = 2; idxIC = 1;}
    else        { idxID = 1; idxIC = 4;}

    int bS = inputShapeInfo[1];                          // batch size
    int iC = inputShapeInfo[idxIC+1];                    // input channels            
    int iD = inputShapeInfo[idxID+1];                    // input depth
    int iH = inputShapeInfo[idxID+2];                    // input height
    int iW = inputShapeInfo[idxID+3];                    // input width

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    outputShapeInfo[0] = 5;
    outputShapeInfo[1] = bS;

    if (isNCDHW) {    
        outputShapeInfo[2] = iC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = iC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew_bp, 2, 1, false, 0, 10) {
    
    NDArray<T> *input = INPUT_VARIABLE(0);                                                  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *gradO = INPUT_VARIABLE(1);                                                  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    
    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM AVGPOOL3DNEW_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
                                     
    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID
    int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // 1-NDHWC, 0-NCDHW    

    int idxID, idxIC;
    if(isNCHW) { idxID = 2; idxIC = 1;}
    else       { idxID = 1; idxIC = 4;}

    int bS = input->sizeAt(0);                  // batch size
    int iC = input->sizeAt(idxIC);              // input channels        
    int iD = input->sizeAt(idxID);              // input depth
    int iH = input->sizeAt(idxID+1);            // input height
    int iW = input->sizeAt(idxID+2);            // input width    
    int oD = gradO->sizeAt(idxID);              // output depth
    int oH = gradO->sizeAt(idxID+1);            // output height
    int oW = gradO->sizeAt(idxID+2);            // output width             

    REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM AVGPOOL3D_BP OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
    REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM AVGPOOL3D_BP OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    
      
    if(!isNCHW) {
        gradO = gradO ->permute({0, 4, 1, 2, 3});                                                       // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
        gradI = new NDArray<T>(gradI->ordering(), {bS, iC, iD, iH, iW}, block.getWorkspace());                                  // [bS, iC, iD, iH, iW]

        gradO->streamline('c');       
    }
   
    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 1, 1, 1);    
    
    ConvolutionUtils<T>::avgPool3DBP(*gradO, *gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, !isSameMode);

    if(!isNCHW) {              

        gradI->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, iD, iH, iW] -> [bS, iD, iH, iW, iC]
        OUTPUT_VARIABLE(0)->assign(gradI);
        
        delete gradO;
        delete gradI;
    }        

    return Status::OK();
}


DECLARE_SHAPE_FN(avgpool3dnew_bp) {

    int* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);
        
    return SHAPELIST(gradIshapeInfo);        
}



}
}

#endif