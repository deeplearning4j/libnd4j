//
// created by Yurii Shyrma on 05.02.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(conv3dNew, 2, 1, false, 0, 12) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                        // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                        // [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;            // [oC]
    NDArray<T> *output  = OUTPUT_VARIABLE(2);                                       // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)
    
    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D OP: rank of input array must be equal to 5 !");
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to 5 !");
    REQUIRE_TRUE(bias->rankOf()    == 1, 0, "CUSTOM CONV3D OP: rank of biases array must be equal to 1 !");
                                     
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
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID;
    int dataFormat  = block.getIArguments()->size() > 13 ? INT_ARG(13) : 0;     // 0-NDHWC, 1-NCDHW
    
    int indID = dataFormat == 0 ? 1 : 2;
    int indIC = dataFormat == 0 ? 4 : 1;    
    int indOC = dataFormat == 0 ? 4 : 0;
    int indWC = dataFormat == 0 ? 3 : 1;
    int indKD = dataFormat == 0 ? 0 : 2;

    int bS = input->sizeAt(0);                 // batch size
    int iD = input->sizeAt(indID);             // input depth
    int iH = input->sizeAt(indID+1);           // input height
    int iW = input->sizeAt(indID+2);           // input width
    int iC = input->sizeAt(indIC);             // input channels        
    int oC = weights->sizeAt(indOC);           // output channels
    
    REQUIRE_TRUE(weights->sizeAt(indWC)   == iC, 0, "CUSTOM CONV3D OP: the wrong shape of weights array, input_inChannels != weights_inChannels");
    REQUIRE_TRUE(weights->sizeAt(indKD)   == kD, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    REQUIRE_TRUE(weights->sizeAt(indKD+1) == kH, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    REQUIRE_TRUE(weights->sizeAt(indKD+2) == kW, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    if (bias)
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP:: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());
        
    int oD = output->sizeAt(indID);               // output depth
    int oH = output->sizeAt(indID+1);             // output height
    int oW = output->sizeAt(indID+2);             // output width
    
    std::vector<int> kDims = {iC*kD*kH*kW, oC};    
    std::vector<int> preContractDims  = {bS*oD*oH*oW, iC*kD*kH*kW};
    // columns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);
    std::vector<int> contractDims     = {1, 0};
    std::vector<int> postContractDims = {bS, oD, oH, oW, oC};

    ResultSet<T>* inSubArrsList  = NDArrayFactory<T>::allExamples(input);           // inSubArrsList.size() = outSubArrsList.size() = bS
    ResultSet<T>* outSubArrsList = NDArrayFactory<T>::allExamples(output);

    for(int i = 0; i < bS; ++i)
        ConvolutionUtils<T>::vol2col(inSubArrsList->at(i)->getBuffer(), oC, oD, oH, oW, iD, iH, iW, kD, kH, kW, pD, pH, pW, sD, sH, sW, dD, dH, dW, outSubArrsList->at(i)->getBuffer());


    delete inSubArrsList;
    delete outSubArrsList;


//      static void _vol2col(const T* data_col, const int channels, const int depth, const int height, const int width, const int out_depth, const int out_height, const int out_width, 
//         const int kT, const int kH, const int kW, 
//         const int pT, const int pH, const int pW, 
//         const int dT, const int dH, const int dW, 
//         const int dilationT, const int dilationH, const int dilationW, 
//         T* data_vol);


// (derived(), patch_planes, patch_rows, patch_cols, plane_stride, row_stride, col_stride, 1, 1, 1, 1, 1, 1, padding_type, padding_value);

            
// input.extract_volume_patches(kD, kH, kW, sD, sH, sW,padding_type).reshape(preContractDims) .contract(kernel.reshape(kDims), contractDims).reshape(postContractDims)


//   THNN_(vol2col)(
//       THTensor_(data)(input_n),
//       nInputPlane, inputDepth, inputHeight, inputWidth,
//       kT, kH, kW, padT, padH, padW, dT, dH, dW,
//       dilationT, dilationH, dilationW,
//       THTensor_(data)(columns)
//     );




// // std::unique_ptr<NDArray<T>> col(new NDArray<T>('c', {batchSize, oY, oX, inDepth, kY, kX}));
// //             std::unique_ptr<NDArray<T>> col2(col.get()->permute({0, 3, 4, 5, 1, 2}));

// //             std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

// //             input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.data());







    
    delete bias;

    
    return Status::OK();
}


DECLARE_SHAPE_FN(conv3dNew) {

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
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID;
    int dataFormat  = block.getIArguments()->size() > 13 ? INT_ARG(13) : 0;     // 0-NDHWC, 1-NCDHW
    
    int indID  = dataFormat == 0 ? 1 : 2;
    int indIC  = dataFormat == 0 ? 4 : 1;
    int indOC = dataFormat == 0 ? 4 : 0;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                         // batch size
    int iD = inputShapeInfo[indID+1];                    // input depth
    int iH = inputShapeInfo[indID+2];                    // input height
    int iW = inputShapeInfo[indID+3];                    // input width
    int iC = inputShapeInfo[indIC+1];                    // input channels        
    int oC = weightsShapeInfo[indOC+1];                 // output channels

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    if(!paddingMode)                        // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);
    outputShapeInfo[0]      = 5;
    outputShapeInfo[1]      = bS;
    outputShapeInfo[indID+1] = oD;
    outputShapeInfo[indID+2] = oH;
    outputShapeInfo[indID+3] = oW;
    outputShapeInfo[indIC+1] = oC;
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));
    
    return new ShapeList(outputShapeInfo);
}


}
}


