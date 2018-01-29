//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(fused_batch_norm, 3, 1, false, 1, 2) {

    NDArray<T>* x      = INPUT_VARIABLE(0);                 // [bS,iH,iW,iD] or [bS,iD,iH,iW]
    NDArray<T>* scale  = INPUT_VARIABLE(1);                 // [iD]
    NDArray<T>* offset = INPUT_VARIABLE(2);                 // [iD]

    NDArray<T>* y = OUTPUT_VARIABLE(0);                     
    NDArray<T>* batchMean = OUTPUT_VARIABLE(1);
    NDArray<T>* batchVar  = OUTPUT_VARIABLE(2);

    const bool dataFormat = (bool)INT_ARG(0);  // 0->NHWC, 1->NCHW
    const bool isTraining = (bool)INT_ARG(1);    


    REQUIRE_TRUE(x->rankOf() == 4, 0, "CUSTOM_OP fused_batch_norm: the rank of input x array must be equal to 4 !");
    REQUIRE_TRUE(scale->rankOf() == 1, 0, "CUSTOM_OP fused_batch_norm: the rank of input scale array must be equal to 1 !");
    REQUIRE_TRUE(offset->rankOf() == 1, 0, "CUSTOM_OP fused_batch_norm: the rank of input offset array must be equal to 1 !");    

    NDArray<T>* xR = x->dup();
    if(dataFormat)
        xR->reshapei(xR->ordering(), {0,2,3,1});

    const int bS = xR->sizeAt(0);           // batch size
    const int iH = xR->sizeAt(1);           // input height
    const int iW = xR->sizeAt(2);           // input width
    const int iD = xR->sizeAt(3);           // input iD or number of channels

    REQUIRE_TRUE(scale->sizeAt(0)  == iD, 0, "CUSTOM_OP fused_batch_norm: the length of input scale array must be number of channels (input iD) !");
    REQUIRE_TRUE(offset->sizeAt(0) == iD, 0, "CUSTOM_OP fused_batch_norm: the length of input offset array must be number of channels (input iD) !");

    NDArray<T>* mean(nullptr), *variance(nullptr);
    if(!isTraining){
        mean     = INPUT_VARIABLE(3);   
        variance = INPUT_VARIABLE(4);   
        REQUIRE_TRUE(mean->rankOf() == 1, 0, "CUSTOM_OP fused_batch_norm: the rank of input mean array must be equal to 1 !");
        REQUIRE_TRUE(variance->rankOf() == 1, 0, "CUSTOM_OP fused_batch_norm: the rank of input variance array must be equal to 1 !");    
        REQUIRE_TRUE(mean->sizeAt(0)  == iD, 0, "CUSTOM_OP fused_batch_norm: the length of input scale array must be number of channels (input iD) !");
        REQUIRE_TRUE(variance->sizeAt(0) == iD, 0, "CUSTOM_OP fused_batch_norm: the length of input variance array must be number of channels (input iD) !");
    }
    else {
        REQUIRE_TRUE(block.width() == 3, 0, "CUSTOM_OP fused_batch_norm: when isTraining=true then number of input arrays must be equal to 3 !");   
        std::vector<int> shape = {iD};
        mean = new NDArray<T>(scale->ordering(), shape, block.getWorkspace());
        variance = new NDArray<T>(scale->ordering(), shape, block.getWorkspace());
    }

    T epsilon;
    if(block.getTArguments()->size() > 0)
        epsilon = T_ARG(0);
    else 
        epsilon = 0.001;
    
    epsilon = epsilon > (T)1.001e-5 ? epsilon : (T)1.001e-5;    
    
    
    const int restSize = xR->lengthOf() / iD;
    xR->reshapei(xR->ordering(), {restSize, iD});

    const int restSizeMinusOne = (restSize > 1) ? (restSize - 1) : 1;
    T restSizeInv = (T)1.0 / restSize;
    T restSizeAdjust = (T)restSize / restSizeMinusOne;

    if(isTraining) {
        NDArray<T>* sum = xR->sum({0});
        mean->assign((*sum) * restSizeInv);
        batchMean->assign(mean);
        delete sum;
    }
    
    *xR -= *mean;

    if(isTraining) {        
        T power = 2.;
        NDArray<T>* sum = xR->template transform<simdOps::Pow<T>>(&power).sum({0});
        variance->assign((*sum) * restSizeInv);
        batchVar->assign(variance);
        delete sum;
    }

    y->assign( (*xR) * (*scale) * (*variance + epsilon).template transform<simdOps::RSqrt<T>>() + (*offset) );

    delete xR;

    if(isTraining) {
        delete mean;
        delete variance;
    }

    return Status::OK();
}



DECLARE_SHAPE_FN(fused_batch_norm) {

    int* xShapeInfo     = inputShape->at(0);
    int* scaleShapeInfo = inputShape->at(1);
    
    int* outShapeInfo(nullptr), *batchMeanShapeInfo(nullptr), *batchVarShapeInfo(nullptr);
    
    COPY_SHAPE(xShapeInfo, outShapeInfo);
    COPY_SHAPE(scaleShapeInfo, batchMeanShapeInfo);
    COPY_SHAPE(scaleShapeInfo, batchVarShapeInfo);
    
    return new ShapeList({outShapeInfo, batchMeanShapeInfo, batchVarShapeInfo});                
}




}
}