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
    
    // evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::isBlock2x2NotDiag(NDArray<T>& block, int p, int q, T& maxElem) {
        
    NDArray<T> rotation(2, 2, _M.ordering(), _M.getWorkspace());    
    T n = math::nd4j_sqrt<T>(block(p,p)*block(p,p) + block(q,p)*block(q,p));

    const T almostZero = DataTypeUtils::min<T>();
    const T precision = DataTypeUtils::eps<T>();

    if(n == (T)0.)      
        block(p,p) = block(q,p) = 0.;
    else {

        rotation(0,0) =  rotation(1,1) = block(p,p) / n;
        rotation(0,1) = block(q,p) / n;
        rotation(1,0) = -rotation(0,1);
        
        IndicesList indices({NDIndex::interval(p, q+1, q-p), NDIndex::all()});
        NDArray<T>* temp = block.subarray(indices);
        temp->assign(mmul(rotation, *temp));
        delete temp;

        if(_calcU) {

            IndicesList indices({NDIndex::all(), NDIndex::interval(p, q+1, q-p)});
            NDArray<T>* temp1 = _U.subarray(indices);
            NDArray<T>* temp2 = rotation.transpose();
            temp1->assign(mmul(*temp1, *temp2));
            delete temp1;
            delete temp2;
        }
    }
    
    maxElem = math::nd4j_max<T>(maxElem, math::nd4j_max<T>(math::nd4j_abs<T>(block(p,p)), math::nd4j_abs<T>(block(q,q))));
    T threshold = math::nd4j_max<T>(almostZero, precision * maxElem);
 
    return math::nd4j_abs<T>(block(p,q)) > threshold || math::nd4j_abs<T>(block(q,p)) > threshold;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::createJacobiRotation(const T& x, const T& y, const T& z, NDArray<T>& rotation) {
  
    T denom = 2.* math::nd4j_abs<T>(y);

    if(denom < DataTypeUtils::min<T>()) {
        
        rotation(0,0) = rotation(1,1) = 1.;
        rotation(0,1) = rotation(1,0) = 0.;
        return false;
    } 
    else {
        
        T tau = (x-z)/denom;
        T w = math::nd4j_sqrt<T>(tau*tau + 1.);
        T t;
  
        if(tau > (T)0.)
            t = 1. / (tau + w);
        else
            t = 1. / (tau - w);
  
        T sign = t > (T)0. ? 1. : -1.;
        T n = 1. / math::nd4j_sqrt<T>(t*t + 1.);
        rotation(0,0) = rotation(1,1) = n;
        rotation(0,1) = -sign * (y / math::nd4j_abs<T>(y)) * math::nd4j_abs<T>(t) * n;
        rotation(1,0) = -rotation(0,1);

        return true;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::svd2x2(const NDArray<T>& block, int p, int q, NDArray<T>& left, NDArray<T>& right) {
    
    NDArray<T> m(block.ordering(), {2,2}, {block(p,p), block(p,q), block(q,p), block(q,q)}, block.getWorkspace());
  
    NDArray<T> rotation(2, 2, block.ordering(), block.getWorkspace());
    T t = m(0,0) + m(1,1);
    T d = m(1,0) - m(0,1);

    if(math::nd4j_abs<T>(d) < DataTypeUtils::min<T>()) {
    
        rotation(0,0) = rotation(1,1) = 1.;
        rotation(0,1) = rotation(1,0) = 0.;
    }
    else {    
    
        T u = t / d;
        T tmp = math::nd4j_sqrt<T>(1. + u*u);
        rotation(0,0) = rotation(1,1) = u / tmp;
        rotation(0,1) = 1./tmp;
        rotation(1,0) = -rotation(0,1);        
    }
              
    m.assign(mmul(rotation, m));
    createJacobiRotation(m(0,0), m(0,1), m(1,1), right);

    NDArray<T> *temp = right.transpose();
    left.assign(mmul(rotation, *temp));
    delete temp;
    
}


//////////////////////////////////////////////////////////////////////////
// template <typename T>
// void JacobiSVD<T>::evalData(const NDArray<T>& matrix) {

//     const T precision  = (T)2. * DataTypeUtils::eps<T>();  
//     const T almostZero = DataTypeUtils::min<T>();

//     T scale = matrix.template reduceNumber<simdOps::AMax<T>>();
//     if(scale== (T)0.) 
//         scale = 1.;

//     if(_rows > _cols) {
        
//         HHcolPivQR<T> qr(matrix / scale);
//         _M.assign(qr._qr({{0, _cols},{0, _cols}}));
//         _M.setZeros("trianLow");
            
//         HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');

//         if(_fullUV)
//             hhSeg.applyTo(_U);             
//         else if(_calcU) {            
//             _U.setIdentity();
//             hhSeg.mulLeft(_U);
//         }
        
//         if(_calcV)
//             _V.assign(qr._permut);
//     }    
//     else if(_rows < _cols) {

//         NDArray<T>* matrixT = matrix.transpose();
//         HHcolPivQR<T> qr(*matrixT / scale);
//         _M.assign(qr._qr({{0, _rows},{0, _rows}}));
//         _M.setZeros("trianLow");
//         _M.transposei();
    
//         HHsequence<T>  hhSeg(qr._qr, qr._coeffs, 'u');          // type = 'u' is not mistake here !

//         if(_fullUV)
//             hhSeg.applyTo(_V);             
//         else if(_calcV) {            
//             _V.setIdentity();
//             hhSeg.mulLeft(_V);        
//         }
                        
//         if(_calcU)
//             _U.assign(qr._permut);
        
//         delete matrixT;
      
//     }
//     else {

//         _M.assign(matrix({{0 _diagSize}, {0,_diagSize}})/scale);

//         if(calcU) 
//             _U.setIdentity();

//         if(calcV) 
//             _V.setIdentity();
//     }

// }




template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

