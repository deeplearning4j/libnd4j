// 
// created by Yurii Shyrma on 05.12.2017
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gru.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell, 5, 1, false, 0, 0) {

    NDArray<T>* x  = INPUT_VARIABLE(0);                     // input [bS x inSize]
    NDArray<T>* h0ShapeInfo = INPUT_VARIABLE(1);                     // previous cell output [bS x numUnits],  that is at previous time step t-1

    NDArray<T>* WxShapeInfo   = INPUT_VARIABLE(2);                   // input-to-hidden weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b    = INPUT_VARIABLE(4);                   // biases, [3*numUnits] 
    
    NDArray<T>* h    =  OUTPUT_VARIABLE(0);                  // current cell output [bS x numUnits], that is at current time step t    

    const int rank     = x->rankOf();              // = 2    
    const int bS       = x->sizeAt(0);
    const int inSize   = x->sizeAt(1);
    const int numUnits = h0ShapeInfo->sizeAt(1);    

    const std::string h0Shape        = ShapeUtils<T>::shapeAsString(h0ShapeInfo); 
    const std::string h0CorrectShape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils<T>::shapeAsString(WxShapeInfo); 
    const std::string wxCorrectShape = ShapeUtils<T>::shapeAsString({inSize, 3*numUnits}); 
    const std::string whShape        = ShapeUtils<T>::shapeAsString(Wh); 
    const std::string whCorrectShape = ShapeUtils<T>::shapeAsString({numUnits, 3*numUnits}); 
    const std::string bShape         = ShapeUtils<T>::shapeAsString(b); 
    const std::string bCorrectShape  = ShapeUtils<T>::shapeAsString({3*numUnits});    
    
    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRUCELL operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str()); 
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRUCELL operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str()); 
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRUCELL operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());     
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRUCELL operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());     

    
    helpers::gruCell({x, h0ShapeInfo, WxShapeInfo, Wh, b}, h);

    return Status::OK();
}



DECLARE_SHAPE_FN(gruCell) {    
    
    const int* xShapeInfo  = inputShape->at(0);                     // input [bS x inSize]
    const int* h0ShapeInfo = inputShape->at(1);                     // previous cell output [bS x numUnits],  that is at previous time step t-1
    const int* WxShapeInfo = inputShape->at(2);                     // input-to-hidden weights, [inSize   x 3*numUnits] 
    const int* WhShapeInfo = inputShape->at(3);                     // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    const int* bShapeInfo  = inputShape->at(4);                     // biases, [3*numUnits] 

    const int rank     = xShapeInfo[0];              // = 2    
    const int bS       = xShapeInfo[1];
    const int inSize   = xShapeInfo[2];
    const int numUnits = h0ShapeInfo[2];    

    const std::string h0Shape        = ShapeUtils<T>::shapeAsString(h0ShapeInfo); 
    const std::string h0CorrectShape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils<T>::shapeAsString(WxShapeInfo); 
    const std::string wxCorrectShape = ShapeUtils<T>::shapeAsString({inSize, 3*numUnits}); 
    const std::string whShape        = ShapeUtils<T>::shapeAsString(WhShapeInfo); 
    const std::string whCorrectShape = ShapeUtils<T>::shapeAsString({numUnits, 3*numUnits}); 
    const std::string bShape         = ShapeUtils<T>::shapeAsString(bShapeInfo); 
    const std::string bCorrectShape  = ShapeUtils<T>::shapeAsString({3*numUnits});    
    
    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRUCELL operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str()); 
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRUCELL operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str()); 
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRUCELL operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());     
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRUCELL operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());     
        
    int* hShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);       // [bS x numUnits]
    
    hShapeInfo[0] = rank;        
    hShapeInfo[1] = bS;
    hShapeInfo[2] = numUnits;
        
    shape::updateStrides(hShapeInfo, shape::order(const_cast<int*>(h0ShapeInfo)));
    
    return SHAPELIST(hShapeInfo);
}   






}
}

