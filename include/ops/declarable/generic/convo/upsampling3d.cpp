//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 04.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_upsampling3d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling3d, 1, 1, false, 0, 3) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);             // [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    NDArray<T>* output = OUTPUT_VARIABLE(0);            // [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
            
    const int factorD = INT_ARG(0);
    const int factorH = INT_ARG(1);
    const int factorW = INT_ARG(2);
    const int isNCDHW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 0;       // 1-NCDHW,  0-NDHWC

    REQUIRE_TRUE(input->rankOf() == 5, 0, "UPSAMPLING3D op: input should be 5D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(output->rankOf() == 5, 0, "UPSAMPLING3D op: output should be 5D, but got %i instead!", output->rankOf());

    ConvolutionUtils<T>::upsampling3d(*input, *output, factorD, factorH, factorW, (bool)isNCDHW);

    return Status::OK();
}

        
DECLARE_SHAPE_FN(upsampling3d) {
    
    int* inputShapeInfo = inputShape->at(0);
    
    REQUIRE_TRUE(inputShapeInfo[0] == 5, 0, "UPSAMPLING2D op: input should be 5D, but got %i instead!", inputShapeInfo[0]);

    const int factorD = INT_ARG(0);
    const int factorH = INT_ARG(1);
    const int factorW = INT_ARG(2);
    const int isNCDHW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 0;       // 1-NCDHW,  0-NDHWC

    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo[0]), int);

    outputShapeInfo[0] = inputShapeInfo[0];
    outputShapeInfo[1] = inputShapeInfo[1];
    
    if(isNCDHW) {
        outputShapeInfo[2] = inputShapeInfo[2];
        outputShapeInfo[3] = inputShapeInfo[3] * factorD;
        outputShapeInfo[4] = inputShapeInfo[4] * factorH;
        outputShapeInfo[5] = inputShapeInfo[5] * factorW;
    }
    else {        
        outputShapeInfo[2] = inputShapeInfo[2] * factorD;
        outputShapeInfo[3] = inputShapeInfo[3] * factorH;
        outputShapeInfo[4] = inputShapeInfo[4] * factorW;
        outputShapeInfo[5] = inputShapeInfo[5];
    }

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}

}
}

#endif