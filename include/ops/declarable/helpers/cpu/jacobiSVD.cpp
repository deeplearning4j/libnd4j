//
// Created by Yurii Shyrma on 11.01.2018
//

#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/hhColPivQR.h>


namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
JacobiSVD<T>::JacobiSVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV) {

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
    
    evalData(matrix);
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::evalData(const NDArray<T>& matrix) {

    const T precision  = (T)2. * DataTypeUtils::eps<T>();  
    const T almostZero = DataTypeUtils::eps<T>();

    T scale = matrix.template reduceNumber<simdOps::AMax<T>>();
    if(scale== (T)0.) 
        scale = 1.;

    if(_rows > _cols) {
        
        HHcolPivQR<T> qr(matrix / scale);
        _M.assign(qr._qr({{0, _cols},{0, _cols}}));
        _M.setZeros("trianLow");
            
        HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');

        if(_fullUV)
            hhSeg.applyTo(_U);             
        else if(_calcU) {            
            _U.setIdentity();
            hhSeg.mulLeft(_U);
        }
        
        if(_calcV)
            _V.assign(qr._permut);
    }    
    else if(_rows < _cols) {

        NDArray<T>* matrixT = matrix.transpose();
        HHcolPivQR<T> qr(*matrixT / scale);
        _M.assign(qr._qr({{0, _rows},{0, _rows}}));
        _M.setZeros("trianLow");
        _M.transposei();
    
        HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');          // type = 'u' is not mistake here !

        if(_fullUV)
            hhSeg.applyTo(_V);             
        else if(_calcV) {            
            _V.setIdentity();
            hhSeg.mulLeft(_V);        
        }
                        
        if(_calcU)
            _U.assign(qr._permut);
        
        delete matrixT;
      
    }
    else {

    }

}




template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

