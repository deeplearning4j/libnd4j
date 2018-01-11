//
// Created by Yurii Shyrma on 11.01.2018
//

#include <ops/declarable/helpers/jacobiSVD.h>


namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::JacobiSVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw "ops::helpers::JacobiSVD constructor: input array must be 2D matrix !";

    _rows = matrix.sizeAt(0);
    _cols = matrix.sizeAt(1);
    _diagSize = math::nd4j_min<int>(_rows, _cols);    

    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;

    _S = NDArray<T>(_diagSize, 1, matrix.ordering(), matrix.getWorkspace());

    if(_calcU) {
        if(_fullUV)
            _U = NDArray<T>(_rows, _rows, matrix.ordering(), matrix.getWorkspace());   
        else
            _U = NDArray<T>(_rows, _diagSize, matrix.ordering(), matrix.getWorkspace());   
    }
    else 
        _U = NDArray<T>(_rows, 1, matrix.ordering(), matrix.getWorkspace());   

    if(_calcV) {
        if(_fullUV)
            _V = NDArray<T>(_cols, _cols, matrix.ordering(), matrix.getWorkspace());   
        else
            _V = NDArray<T>(_cols, _diagSize, matrix.ordering(), matrix.getWorkspace());   
    }
    else 
        _V = NDArray<T>(_cols, 1, matrix.ordering(), matrix.getWorkspace());   
    
    _M = NDArray<T>(_diagSize, _diagSize, matrix.ordering(), matrix.getWorkspace());
    

}




template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

