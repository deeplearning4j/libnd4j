//
// Created by Yurii Shyrma on 06.12.2017.
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
// Construct an identity matrix, or a batch of matrices
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/eye
CUSTOM_OP_IMPL(diag, 1, 1, false, 0, 0) {

	NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    

    ResultSet<T>* tads = NDArrayFactory<T>::allTensorsAlongDimension(output, {-1,-2});
        
    for(int i = 0; i < tads->size(); ++i) {
        NDArray<T>* arr =  tads->at(i);
        for(int row = 0; row < arr->sizeAt(0); ++row)
            for(int col = 0; col < arr->sizeAt(1); ++col)
                if(row == col)
        
        
    }
	delete tads;

    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(diag) {

    NDArray<T>* input = INPUT_VARIABLE(0);

    const int rank = input->rankOf();

    if(rank > 3)
        throw "CUSTOM_OP diag: rank of input array must be <= 3 !";

    int* outputShapeInfo = nullptr;

    if(input->isVector() || input->isScalar()) {
        ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);
        outputShapeInfo[0] = rank;
        outputShapeInfo[1] = outputShapeInfo[2] = input->lengthOf();
    }
    else {
        ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);
        outputShapeInfo[0] = 2*rank;
        for(int i = 0; i < rank; ++i)
            outputShapeInfo[i + 1] = outputShapeInfo[i + 1 + rank] = input->sizeAt(i);
    }
    	
    shape::updateStrides(outputShapeInfo, input->ordering());    

    return new ShapeList(outputShapeInfo);
    
}





}
}