//
// Created by Yurii Shyrma on 03.01.2018
//

#include <ops/declarable/helpers/svd.h>


namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw "ops::helpers::SVD constructor: input array must be 2D matrix !";

    const int rows = matrix.sizeAt(0);
    const int cols = matrix.sizeAt(1);
    
    if(cols > rows) {

        _transp = true;
        _diagSize = rows;
    }
    else {

        _transp = false;
        _diagSize = cols;
    }

    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;

    if (_transp)
        nd4j::math::nd4j_swap<bool>(_calcU, _calcV);

    _M = NDArray<T>(_diagSize + 1, _diagSize, matrix.ordering(), matrix.getWorkspace());
    _M.assign(0.);     
  
    if (_calcU)
        _U = NDArray<T>(_diagSize + 1, _diagSize + 1, matrix.ordering(), matrix.getWorkspace());
    else         
        _U = NDArray<T>(2, _diagSize + 1, matrix.ordering(), matrix.getWorkspace());
    _U.assign(0.);
      
    if (_calcV) {

        _V = NDArray<T>(_diagSize, _diagSize, matrix.ordering(), matrix.getWorkspace());    
        _V.assign(0.);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation1(int col1, int shift, int ind, int size) {

    if(ind <= 0)
        throw "ops::helpers::SVD::deflation1 method: input index must satisfy condition ind > 0 !";

    int first = col1 + shift;    
    T cos = _M(first, first);
    T sin = _M(first+ind, first);
    T denom = nd4j::math::nd4j_sqrt(cos*cos + sin*sin);

    if (denom == (T)0.) {
        
        _M(first+ind, first+ind) = 0.;
        return;
    }

    cos /= denom;
    sin /= denom;

    _M(first,first) = denom;  
    _M(first+ind, first) = 0.;
    _M(first+ind, first+ind) = 0.;
    
    NDArray<T> rotation(_M.ordering(), {2,2}, {cos, -sin, sin, cos}, _M.getWorkspace());

    NDArray<T>* temp(nullptr);
    
    if (_calcU) {
        
        IndicesList indices({NDIndex::interval(col1, col1 + size + 1), NDIndex::interval(col1, col1 + ind + 1, ind)});
        temp = _U.subarray(indices);
    }
    else {
        
        IndicesList indices({NDIndex::all(), NDIndex::interval(col1, col1 + ind + 1, ind)});
        temp = _U.subarray(indices);
    }

    temp->assign(mmul(*temp, rotation));
    
    delete temp;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation2(int col1U , int col1M, int row1W, int col1W, int ind1, int ind2, int size) {
  
    if(ind1 >= ind2)
        throw "ops::helpers::SVD::deflation2 method: input indexes must satisfy condition ind1 < ind2 !";

    if(size <= 0)
        throw "ops::helpers::SVD::deflation2 method: input size must satisfy condition size > 0 !";

    T cos = _M(col1M+ind1, col1M);
    T sin = _M(col1M+ind2, col1M);  
    T denom = nd4j::math::nd4j_sqrt(cos*cos + sin*sin);
    
    if (denom == (T)0.)  {
      
      _M(col1M + ind1, col1M + ind1) = _M(col1M + ind2, col1M + ind2);
      return;
    }

    cos /= denom;
    sin /= denom;
    _M(col1M + ind1, col1M) = denom;  
    _M(col1M + ind2, col1M + ind2) = _M(col1M + ind1, col1M + ind1);
    _M(col1M + ind2, col1M) = 0.;
    
    NDArray<T> rotation(_M.ordering(), {2,2}, {cos, -sin, sin, cos}, _M.getWorkspace());

    NDArray<T>* temp(nullptr);
    if (_calcU) {
        
        IndicesList indices({NDIndex::interval(col1U, col1U + size + 1), NDIndex::interval(col1U + ind1, col1U + ind2 + 1, ind2 - ind1)});
        temp = _U.subarray(indices);        
    }
    else {

        IndicesList indices({NDIndex::all(), NDIndex::interval(col1U + ind1, col1U + ind2 + 1, ind2 - ind1)});
        temp = _U.subarray(indices);        
    }

    temp->assign(mmul(*temp, rotation));
    delete temp;
    
    if (_calcV)  {
        
        IndicesList indices({NDIndex::interval(row1W, row1W + size), NDIndex::interval(col1W + ind1, col1W + ind2 + 1, ind2 - ind1)});        
        temp = _V.subarray(indices);        
        temp->assign(mmul(*temp, rotation));
        delete temp;
    }
}


template class ND4J_EXPORT SVD<float>;
template class ND4J_EXPORT SVD<float16>;
template class ND4J_EXPORT SVD<double>;







}
}
}
