//
//  @author Yurii Shyrma, created on 05.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sruCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/sru.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sruCell, 4, 2, false, 0, 0) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);               // input [bS x inSize], bS - batch size, inSize - number of features
    NDArray<T>* ct_1 = INPUT_VARIABLE(1);               // previous cell state ct  [bS x inSize], that is at previous time step t-1   
    NDArray<T>* w    = INPUT_VARIABLE(2);               // weights [inSize x 3*inSize]
    NDArray<T>* b    = INPUT_VARIABLE(3);               // biases [1 × 2*inSize]

    NDArray<T>* ht   = OUTPUT_VARIABLE(0);              // current cell output [bS x inSize], that is at current time step t
    NDArray<T>* ct   = OUTPUT_VARIABLE(1);              // current cell state  [bS x inSize], that is at current time step t

    const int rank   = xt->rankOf();
    const int bS     = xt->sizeAt(0);    
    const int inSize = xt->sizeAt(1);                   // inSize - number of features

    // input shapes validation
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string WShape           = ShapeUtils<T>::shapeAsString(w); 
    const std::string correctWShape    = ShapeUtils<T>::shapeAsString({inSize, 3*inSize});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(b); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({2*inSize});

    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "SRUCELL operation: wrong shape of previous cell state, expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWShape    == WShape,    0, "SRUCELL operation: wrong shape of weights, expected is %s, but got %s instead !", correctWShape.c_str(), WShape.c_str()); 
    REQUIRE_TRUE(correctBShape    == bShape,    0, "SRUCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str()); 

            
    helpers::sruCell<T>({xt, ct_1, w, b}, {ht, ct});
    
    return Status::OK();
}


DECLARE_SHAPE_FN(sruCell) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);               // input [bS x inSize], bS - batch size, inSize - number of features
    NDArray<T>* ct_1 = INPUT_VARIABLE(1);               // previous cell state ct  [bS x inSize], that is at previous time step t-1   
    NDArray<T>* w    = INPUT_VARIABLE(2);               // weights [inSize x 3*inSize]
    NDArray<T>* b    = INPUT_VARIABLE(3);               // biases [2*inSize]

    const int rank   = xt->rankOf();
    const int bS     = xt->sizeAt(0);    
    const int inSize = xt->sizeAt(1);                   // inSize - number of features

    // input shapes validation
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string WShape           = ShapeUtils<T>::shapeAsString(w); 
    const std::string correctWShape    = ShapeUtils<T>::shapeAsString({inSize, 3*inSize});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(b); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({2*inSize});

    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "SRUCELL operation: wrong shape of previous cell state, expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWShape    == WShape,    0, "SRUCELL operation: wrong shape of weights, expected is %s, but got %s instead !", correctWShape.c_str(), WShape.c_str()); 
    REQUIRE_TRUE(correctBShape    == bShape,    0, "SRUCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str()); 
    
    // evaluate output shapeInfos
    int *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);      // [bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);      // [bS x numUnits]
            
    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = bS;
    hShapeInfo[2] = cShapeInfo[2] = inSize;
    
    shape::updateStrides(hShapeInfo, ct_1->ordering());
    shape::updateStrides(cShapeInfo, ct_1->ordering());
         
    return SHAPELIST(hShapeInfo, cShapeInfo);
}   




}
}

#endif