//
// @author Yurii Shyrma, created on 02.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/rnn.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(static_rnn, 4, 2, false, 0, 0) {

    NDArray<T>* x  = INPUT_VARIABLE(0);               // input [time x bS x inSize]
	NDArray<T>* Wx = INPUT_VARIABLE(1);               // input-to-hidden  weights, [inSize  x numUnits] 	
    NDArray<T>* Wh = INPUT_VARIABLE(2);               // hidden-to-hidden weights, [numUnits x numUnits]         
	NDArray<T>* b  = INPUT_VARIABLE(3);               // biases for, [2*numUnits] 

	NDArray<T>* h0          = nullptr;     		      // initial cell output (at time step = 0) [bS x numUnits]	
	NDArray<T>* maxTimeStep = nullptr;			      // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    if(block.width() == 5) {
        if ((*INPUT_VARIABLE(4)).rankOf() == 2)
            h0 = INPUT_VARIABLE(4);
        else
            maxTimeStep = INPUT_VARIABLE(4);
    }
	else if(block.width() == 6) {
        h0 = INPUT_VARIABLE(4);
        maxTimeStep = INPUT_VARIABLE(5);
    }    
    
    NDArray<T>* h      =  OUTPUT_VARIABLE(0);           // cell outputs [time x bS x numUnits]
    NDArray<T>* hFinal =  OUTPUT_VARIABLE(1);           // at the end it will store cell final non-zero output [bS x numUnits]

	REQUIRE_TRUE(x->rankOf() == 3, 0, "STATIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(Wx->rankOf() == 2, 0, "STATIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", Wx->rankOf());    

    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);
    const int inSize   = x->sizeAt(2);
    const int numUnits = Wx->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(Wh) == ShapeUtils<T>::shapeAsString({numUnits, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(Wh).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(b)  == ShapeUtils<T>::shapeAsString({2*numUnits}), 0, "STATIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnits}).c_str(), ShapeUtils<T>::shapeAsString(b).c_str()); 
    if(h0)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0) == ShapeUtils<T>::shapeAsString({bS, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(h0).c_str()); 
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStep)  == ShapeUtils<T>::shapeAsString({bS}), 0, "STATIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({bS}).c_str(), ShapeUtils<T>::shapeAsString(maxTimeStep).c_str()); 


    helpers::rnnTimeLoop<T>({x, Wx, Wh, b, h0, maxTimeStep}, h, hFinal);    
    
    return Status::OK();
}



DECLARE_SHAPE_FN(static_rnn) {    

    int* xShapeInfo  = inputShape->at(0);               // input [time x bS x inSize]
    int* WxShapeInfo = inputShape->at(1);               // input-to-hidden  weights, [inSize  x numUnits]     
    int* WhShapeInfo = inputShape->at(2);               // hidden-to-hidden weights, [numUnits x numUnits]         
    int* bShapeInfo  = inputShape->at(3);               // biases for, [2*numUnits] 

    int* h0ShapeInfo          = nullptr;                // initial cell output (at time step = 0) [bS x numUnits] 
    int* maxTimeStepShapeInfo = nullptr;                // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    if(block.width() == 5) {
        if (inputShape->at(4)[0] == 2)
            h0ShapeInfo = inputShape->at(4);
        else
            maxTimeStepShapeInfo = inputShape->at(4);
    }
    else if(block.width() == 6) {
        h0ShapeInfo = inputShape->at(4);
        maxTimeStepShapeInfo = inputShape->at(5);
    }    

    REQUIRE_TRUE(xShapeInfo[0]  == 3, 0, "STATIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", xShapeInfo[0]);
    REQUIRE_TRUE(WxShapeInfo[0] == 2, 0, "STATIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", WxShapeInfo[0]);    

    const int inRank   = xShapeInfo[0];
    const int time     = xShapeInfo[1];
    const int bS       = xShapeInfo[2];
    const int numUnits = WxShapeInfo[2];

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhShapeInfo) == ShapeUtils<T>::shapeAsString({numUnits, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(WhShapeInfo).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bShapeInfo)  == ShapeUtils<T>::shapeAsString({2*numUnits}), 0, "STATIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnits}).c_str(), ShapeUtils<T>::shapeAsString(bShapeInfo).c_str()); 
    if(h0ShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0ShapeInfo) == ShapeUtils<T>::shapeAsString({bS, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(h0ShapeInfo).c_str()); 
    if(maxTimeStepShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo)  == ShapeUtils<T>::shapeAsString({bS}), 0, "STATIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({bS}).c_str(), ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo).c_str()); 

    // evaluate output shapeInfos
    int *hShapeInfo(nullptr), *hPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,     block.getWorkspace(), shape::shapeInfoLength(inRank), int);
    ALLOCATE(hPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), int);
            
    hShapeInfo[0]     = inRank;
    hPrevShapeInfo[0] = inRank-1;
    hShapeInfo[1]     = time;
    hShapeInfo[2]     = hPrevShapeInfo[1] = bS;
    hShapeInfo[3]     = hPrevShapeInfo[2] = numUnits;

    shape::updateStrides(hShapeInfo,     shape::order(xShapeInfo));    
    shape::updateStrides(hPrevShapeInfo, shape::order(xShapeInfo));
         
    return SHAPELIST(hShapeInfo, hPrevShapeInfo);
}   




}
}
