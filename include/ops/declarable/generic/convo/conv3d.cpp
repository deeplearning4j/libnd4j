//
// created by Yurii Shyrma on 05.02.2018
//


#include <ops/declarable/CustomOperations.h>
// #include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(conv3d, 3, 1, false, 0, 1) {
    
    NDArray<T> *input     = INPUT_VARIABLE(0);      // [bS, iD, iH, iW, iC] or [bS, iC, iD, iH, iW]
    NDArray<T> *filter    = INPUT_VARIABLE(1);      // [kD, kH, kW, iC, oC]
    NDArray<T> *strides   = INPUT_VARIABLE(2);      // [5]
    NDArray<T> *output    = OUTPUT_VARIABLE(2);     // [bS, oD, oH, oW, oC] or [bS, oC, oD, oH, oW]

    // NDArray<T> *dilations = nullptr;                // [5]
    // if (block.width() > 3)
    //     dilations = INPUT_VARIABLE(3);
    // else
    //     dilations = new NDArray<T>(input->ordering(), {5}, {1,1,1,1,1});
    NDArray<T> *dilations = new NDArray<T>(input->ordering(), {5}, {1,1,1,1,1});

    int padding    = INT_ARG(0);                                            // 0-SAME,  1-VALID;
    int dataFormat = block.getTArguments()->size() > 1 ? T_ARG(1) : 0;      // 0-NDHWC, 1-NCDHW     

    REQUIRE_TRUE(input->rankOf()  == 5, 0, "CONFIGURABLE CONV3D OP: rank of input array must be equal to 5 !");
    REQUIRE_TRUE(filter->rankOf() == 5, 0, "CONFIGURABLE CONV3D OP: rank of filter array must be equal to 5 !");
    REQUIRE_TRUE(strides->rankOf()   == 1 && strides->lengthOf()   == 5, 0, "CONFIGURABLE CONV3D OP: rank of strides array must be equal to 1 and length to 5 !");
    REQUIRE_TRUE(dilations->rankOf() == 1 && dilations->lengthOf() == 5, 0, "CONFIGURABLE CONV3D OP: rank of dilations array must be equal to 1 and length to 5 !");
    
    int indD = dataFormat == 0 ? 1 : 2;
    int indC = dataFormat == 0 ? 4 : 1;

    int bS = input  ->sizeAt(0);                 // batch size
    int iD = input  ->sizeAt(indD);              // input depth
    int iH = input  ->sizeAt(indD+1);            // input height
    int iW = input  ->sizeAt(indD+2);            // input width
    int iC = input  ->sizeAt(indC);              // input channels        
    int kD = filter ->sizeAt(0);                 // filter(kernel) depth
    int kH = filter ->sizeAt(1);                 // filter(kernel) height
    int kW = filter ->sizeAt(2);                 // filter(kernel) width
    int oC = filter ->sizeAt(4);                 // output channels
    int sD = strides->sizeAt(indD);              // strides depth
    int sH = strides->sizeAt(indD+1);            // strides height
    int sW = strides->sizeAt(indD+2);            // strides width

    REQUIRE_TRUE(filter->sizeAt(3)     == iC, 0, "CONFIGURABLE CONV3D OP: the third dimension of filter array must be equal to the fourth (or first in case of NCDHW) dimension of input array !");
    REQUIRE_TRUE(strides->sizeAt(0)    == 1,  0, "CONFIGURABLE CONV3D OP: there must be condition strides[0] == 1, current implementation does not yet support other cases !");    
    REQUIRE_TRUE(strides->sizeAt(indC) == 1,  0, "CONFIGURABLE CONV3D OP: there must be condition strides[indC] == 1, current implementation does not yet support other cases !");
    
    int oD = output->sizeAt(indD);               // output depth
    int oH = output->sizeAt(indD+1);             // output height
    int oW = output->sizeAt(indD+2);             // output width
    
    std::vector<int> kDims = {iC*kD*kH*kW, oC};    
    std::vector<int> preContractDims  = {bS*oD*oH*oW, iC*kD*kH*kW};
    std::vector<int> contractDims     = {1, 0};
    std::vector<int> postContractDims = {bS, oD, oH, oW, oC};


choose( Cond<internal::traits<Input>::Layout == ColMajor>(),
           
kernel.reshape(kernel_dims).contract(input.extract_volume_patches(kD, kH, kW, sD,sH, sW, padding_type).reshape(pre_contract_dims),contract_dims).reshape(post_contract_dims),
            
input.extract_volume_patches(kD, kH, kW, sD, sH, sW,padding_type).reshape(pre_contract_dims) .contract(kernel.reshape(kernel_dims), contract_dims).reshape(post_contract_dims)
);














    // if (block.width() > 3)
    delete dilations;

    return Status::OK();
}


DECLARE_SHAPE_FN(conv3d) {

    int padding    = INT_ARG(0);
    int dataFormat = block.getTArguments()->size() > 1 ? T_ARG(1) : 0;      // 0-NDHWC, 1-NCDHW     

    int* inputShapeInfo   = inputShape->at(0);
    int* filterShapeInfo  = inputShape->at(1);
    int* stridesShapeInfo = inputShape->at(2);

    int indD = dataFormat == 0 ? 1 : 2;
    int indC = dataFormat == 0 ? 4 : 1;    

    int bS = inputShapeInfo[1];                  // batch size
    int iD = inputShapeInfo[indD+1];             // input depth
    int iH = inputShapeInfo[indD+2];             // input height
    int iW = inputShapeInfo[indD+3];             // input width
    int kD = filterShapeInfo[1];                 // filter(kernel) depth
    int kH = filterShapeInfo[2];                 // filter(kernel) height
    int kW = filterShapeInfo[3];                 // filter(kernel) width
    int oC = filterShapeInfo[5];                 // output channels
    int sD = stridesShapeInfo[indD+1];           // strides depth
    int sH = stridesShapeInfo[indD+2];           // strides height
    int sW = stridesShapeInfo[indD+3];           // strides width

    int oD, oH, oW;                     // output depth, height, width
    if(padding) {                       // VALID
        oD = (iD - kD + sD) / sD;
        oH = (iH - kH + sH) / sH;
        oW = (iW - kW + sW) / sW;
    }
    else {                              // SAME        
        oD = (iD + sD - 1) / sD;
        oH = (iH + sH - 1) / sH;
        oW = (iW + sW - 1) / sW;
    }    

    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);
    outputShapeInfo[0]      = 5;
    outputShapeInfo[1]      = bS;
    outputShapeInfo[indD+1] = oD;
    outputShapeInfo[indD+2] = oH;
    outputShapeInfo[indD+3] = oW;
    outputShapeInfo[indC+1] = oC;
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));
    
    return new ShapeList(outputShapeInfo);
}


}
}


