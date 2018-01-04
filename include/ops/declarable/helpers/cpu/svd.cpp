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
    T denom = nd4j::math::nd4j_sqrt<T>(cos*cos + sin*sin);

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
    T denom = nd4j::math::nd4j_sqrt<T>(cos*cos + sin*sin);
    
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

//////////////////////////////////////////////////////////////////////////
// has effect on block from (col1+shift, col1+shift) to (col2+shift, col2+shift) inclusively 
template <typename T>
void SVD<T>::deflation(int col1, int col2, int ind, int row1W, int col1W, int shift)
{
    
    const int len = col2 + 1 - col1;

    NDArray<T>* colVec0 = _M.subarray({{col1+shift, col1+shift+len },{col1+shift, col1+shift+1}});  
        
    NDArray<T> diagInterval(len, 1, _M.ordering(), _M.getWorkspace());
    for(int i = 0; i < len; ++i)
        diagInterval(i) = _M(col1+shift+i,col1+shift+i);
  
    const T almostZero = DataTypeUtils::min<T>();
    T maxElem;
    if(len == 1)
        maxElem = nd4j::math::nd4j_abs<T>(diagInterval(0));
    else
        maxElem = diagInterval({{1,-1},{}}).template reduceNumber<simdOps::AMax<T>>();                
    T maxElem0 = colVec0->template reduceNumber<simdOps::AMax<T>>();     

    T eps = nd4j::math::nd4j_max<T>(almostZero, DataTypeUtils::eps<T>() * maxElem);
    T epsBig = (T)8. * DataTypeUtils::eps<T>() * nd4j::math::nd4j_max<T>(maxElem0, maxElem);        

    if(diagInterval(0) < epsBig)
        diagInterval(0) = epsBig;
  
    for(int i=1; i < len; ++i)
        if(nd4j::math::nd4j_abs<T>((*colVec0)(i)) < eps)    
            (*colVec0)(i) = 0.;

    for(int i=1; i < len; i++)
        if(diagInterval(i) < epsBig) {
            deflation1(col1, shift, i, len);    
            for(int i = 0; i < len; ++i)
                diagInterval(i) = _M(col1+shift+i,col1+shift+i);
        }
    
    {
        
        bool totDefl = true;    
        for(int i=1; i < len; i++)
            if((*colVec0)(i) >= almostZero) {
                totDefl = false;
                break;
            }
        diagInterval.printBuffer();
        int* permut = nullptr;    
        ALLOCATE(permut, _M.getWorkspace(), 3*_diagSize, int);
        {
            permut[0] = 0;
            int p = 1;          
            
            for(int i=1; i<len; ++i)
                if(nd4j::math::nd4j_abs<T>(diagInterval(i)) < almostZero)
                    permut[p++] = i;            
            
            int k = 1, m = ind+1;
            
            for( ; p < len; ++p) {
                if(k > ind)             
                    permut[p] = m++;                
                else if(m >= len)
                    permut[p] = k++;
                else if(diagInterval(k) < diagInterval(m))
                    permut[p] = m++;
                else                        
                    permut[p] = k++;
            }
        }
    
        if(totDefl) {
            for(int i=1; i<len; ++i) {
                int ki = permut[i];
                if(nd4j::math::nd4j_abs<T>(diagInterval(ki)) < almostZero || diagInterval(0) < diagInterval(ki))
                    permut[i-1] = permut[i];
                else {
                    permut[i-1] = 0;
                    break;
                }
            }
        }
        
        int *tInd = permut + len;
        int *tCol = permut + 2*len;
    
        for(int m = 0; m < len; m++) {
            tCol[m] = m;
            tInd[m] = m;
        }
    
        for(int i = totDefl ? 0 : 1; i < len; i++) {
        
            const int ki = permut[len - (totDefl ? i+1 : i)];
            const int jac = tCol[ki];
           
            nd4j::math::nd4j_swap<T>(diagInterval(i), diagInterval(jac));
            if(i!=0 && jac!=0) 
            nd4j::math::nd4j_swap<T>((*colVec0)(i), (*colVec0)(jac));
      
            NDArray<T>* temp1 = nullptr, *temp2 = nullptr;
            if (_calcU) {
                temp1 = _U.subarray({{col1, col1+len+1},{col1+i,   col1+i+1}});
                temp2 = _U.subarray({{col1, col1+len+1},{col1+jac, col1+jac+1}});               
            }        
            else {
                temp1 = _U.subarray({{0, 2},{col1+i,   col1+i+1}});
                temp2 = _U.subarray({{0, 2},{col1+jac, col1+jac+1}});                
            }
            temp1->swapUnsafe(*temp2);
            delete temp1;
            delete temp2;

            if(_calcV) {
                temp1 = _U.subarray({{row1W, row1W+len},{col1W+i,   col1W+i+1}});
                temp2 = _U.subarray({{row1W, row1W+len},{col1W+jac, col1W+jac+1}});               
                temp1->swapUnsafe(*temp2);
                delete temp1;
                delete temp2;                
            }
      
            const int tI = tInd[i];
            tCol[tI] = jac;
            tCol[ki] = i;
            tInd[jac] = tI;
            tInd[i] = ki;
        }
        
        RELEASE(permut, _M.getWorkspace());
    }
    
    {
        int i = len-1;
        
        while(i > 0 && (nd4j::math::nd4j_abs<T>(diagInterval(i)) < almostZero || nd4j::math::nd4j_abs<T>((*colVec0)(i)) < almostZero)) 
            --i;
        
        for(; i > 1; --i) {
            if( (diagInterval(i) - diagInterval(i-1)) < DataTypeUtils::eps<T>()*maxElem ) {
                if (nd4j::math::nd4j_abs<T>(diagInterval(i) - diagInterval(i-1)) >= epsBig)                     
                    throw "ops::helpers::SVD::deflation: diagonal elements are not properly sorted !";
                deflation2(col1, col1 + shift, row1W, col1W, i-1, i, len);
            }
        }
    }  

    for(int i = 0; i < len; ++i)
        _M(col1+shift+i,col1+shift+i) = diagInterval(i);    

    delete colVec0;

}






template class ND4J_EXPORT SVD<float>;
template class ND4J_EXPORT SVD<float16>;
template class ND4J_EXPORT SVD<double>;







}
}
}

