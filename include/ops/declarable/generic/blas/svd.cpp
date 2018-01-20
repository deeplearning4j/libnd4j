//
//  // Created by Yurii Shyrma on 20.01.2018
//

#include <ops/declarable/svd.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(svd, 1, 1, false, 0, 3) {
    
    NDArray<T>* x = INPUT_VARIABLE(0);
    NDArray<T>* s = OUTPUT_VARIABLE(0);
    
    const bool fullUV = (bool)INT_ARG(0);
    const bool calcUV = (bool)INT_ARG(1);
    const int switchNum =  INT_ARG(2);    
    const int rank =  x->rankOf();

    ResultSet<T>* listX = NDArrayFactory<T>::allTensorsAlongDimension(x, {rank-2, rank-1});
    ResultSet<T>* listS = NDArrayFactory<T>::allTensorsAlongDimension(s, {rank-1});
    ResultSet<T>* listU(nullptr), *listV(nullptr);
    
    if(calcUV) {
        NDArray<T>* u = OUTPUT_VARIABLE(1);
        NDArray<T>* v = OUTPUT_VARIABLE(2);
        listU = NDArrayFactory<T>::allTensorsAlongDimension(u, {rank-2, rank-1});
        listV = NDArrayFactory<T>::allTensorsAlongDimension(v, {rank-2, rank-1});
    }


    for(int i=0; i<listX->size(); ++i) {
        
        helpers::SVD<T> svd(*(listX->at(i)), switchNum, calcUV, calcUV, fullUV);    
        listS->at(i)->assign(svd._S);

        if(calcUV) {
            listU->at(i)->assign(svd._U);
            listV->at(i)->assign(svd._V);
        }        
    }

    delete listX;
    delete listS;
    
    if(calcUV) {
        delete listU;
        delete listV;
    }
         
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(svd) {

    int* inShapeInfo = inputShape->at(0);
    bool fullUV = (bool)INT_ARG(0);
    bool calcUV = (bool)INT_ARG(1);
    
    const int rank = inShapeInfo[0];
    const int diagSize = inShapeInfo[rank] < inShapeInfo[rank-1] ? inShapeInfo[rank] : inShapeInfo[rank-1];
    
    ALLOCATE(sShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank-1), int); 
    sShapeInfo[0] = rank - 1;
    for(int i=1; i <= rank-2; ++i)
        sShapeInfo[i] = inShapeInfo[i];
    sShapeInfo[rank-1] = diagSize;
    shape::updateStrides(sShapeInfo, shape::order(inShapeInfo));
    
    if(calcUV){

        int* uShapeInfo(nullptr), *vShapeInfo(nullptr);
        COPY_SHAPE(inShapeInfo, uShapeInfo);
        COPY_SHAPE(inShapeInfo, vShapeInfo);
    
        if(fullUV) {
            uShapeInfo[rank]   = uShapeInfo[rank-1];
            vShapeInfo[rank-1] = vShapeInfo[rank];
        }
        else {
            uShapeInfo[rank] = diagSize;
            vShapeInfo[rank] = diagSize;
        }
    
        shape::updateStrides(uShapeInfo, shape::order(inShapeInfo));
        shape::updateStrides(vShapeInfo, shape::order(inShapeInfo));
    
        return new ShapeList({sShapeInfo, uShapeInfo, vShapeInfo});        
    }     
    
    return new ShapeList(sShapeInfo);
}


}
}