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
        math::nd4j_swap<bool>(_calcU, _calcV);

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
        throw "ops::helpers::SVD::deflation1 method: input int must satisfy condition ind > 0 !";

    int first = col1 + shift;    
    T cos = _M(first, first);
    T sin = _M(first+ind, first);
    T denom = math::nd4j_sqrt<T>(cos*cos + sin*sin);

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
        throw "ops::helpers::SVD::deflation2 method: input intes must satisfy condition ind1 < ind2 !";

    if(size <= 0)
        throw "ops::helpers::SVD::deflation2 method: input size must satisfy condition size > 0 !";

    T cos = _M(col1M+ind1, col1M);
    T sin = _M(col1M+ind2, col1M);  
    T denom = math::nd4j_sqrt<T>(cos*cos + sin*sin);
    
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
        
    NDArray<T>* diagInterval = _M({{col1+shift, col1+shift+len},{col1+shift, col1+shift+len}}).diagonal('c');        
  
    const T almostZero = DataTypeUtils::min<T>();
    T maxElem;
    if(len == 1)
        maxElem = math::nd4j_abs<T>((*diagInterval)(0));
    else
        maxElem = (*diagInterval)({{1,-1},{}}).template reduceNumber<simdOps::AMax<T>>();                
    T maxElem0 = colVec0->template reduceNumber<simdOps::AMax<T>>();     

    T eps = math::nd4j_max<T>(almostZero, DataTypeUtils::eps<T>() * maxElem);
    T epsBig = (T)8. * DataTypeUtils::eps<T>() * math::nd4j_max<T>(maxElem0, maxElem);        

    if((*diagInterval)(0) < epsBig)
        (*diagInterval)(0) = epsBig;
  
    for(int i=1; i < len; ++i)
        if(math::nd4j_abs<T>((*colVec0)(i)) < eps)    
            (*colVec0)(i) = 0.;

    for(int i=1; i < len; i++)
        if((*diagInterval)(i) < epsBig) {
            deflation1(col1, shift, i, len);    
            for(int i = 0; i < len; ++i)
                (*diagInterval)(i) = _M(col1+shift+i,col1+shift+i);
        }
    
    {
        
        bool totDefl = true;    
        for(int i=1; i < len; i++)
            if((*colVec0)(i) >= almostZero) {
                totDefl = false;
                break;
            }
        
        int* permut = nullptr;    
        ALLOCATE(permut, _M.getWorkspace(), 3*_diagSize, int);
        {
            permut[0] = 0;
            int p = 1;          
            
            for(int i=1; i<len; ++i)
                if(math::nd4j_abs<T>((*diagInterval)(i)) < almostZero)
                    permut[p++] = i;            
            
            int k = 1, m = ind+1;
            
            for( ; p < len; ++p) {
                if(k > ind)             
                    permut[p] = m++;                
                else if(m >= len)
                    permut[p] = k++;
                else if((*diagInterval)(k) < (*diagInterval)(m))
                    permut[p] = m++;
                else                        
                    permut[p] = k++;
            }
        }
    
        if(totDefl) {
            for(int i=1; i<len; ++i) {
                int ki = permut[i];
                if(math::nd4j_abs<T>((*diagInterval)(ki)) < almostZero || (*diagInterval)(0) < (*diagInterval)(ki))
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
           
            math::nd4j_swap<T>((*diagInterval)(i), (*diagInterval)(jac));
            if(i!=0 && jac!=0) 
            math::nd4j_swap<T>((*colVec0)(i), (*colVec0)(jac));
      
            NDArray<T>* temp1 = nullptr, *temp2 = nullptr;
            if (_calcU) {
                temp1 = _U.subarray({{col1, col1+len+1},{col1+i,   col1+i+1}});
                temp2 = _U.subarray({{col1, col1+len+1},{col1+jac, col1+jac+1}});                     
                NDArray<T> temp3 = *temp1;                
                temp1->assign(temp2);
                temp2->assign(temp3);                
            }        
            else {
                temp1 = _U.subarray({{0, 2},{col1+i,   col1+i+1}});
                temp2 = _U.subarray({{0, 2},{col1+jac, col1+jac+1}});                
                NDArray<T> temp3 = *temp1;                
                temp1->assign(temp2);
                temp2->assign(temp3);                
            }            
            delete temp1;
            delete temp2;

            if(_calcV) {
                temp1 = _V.subarray({{row1W, row1W+len},{col1W+i,   col1W+i+1}});
                temp2 = _V.subarray({{row1W, row1W+len},{col1W+jac, col1W+jac+1}});               
                NDArray<T> temp3 = *temp1;                
                temp1->assign(temp2);
                temp2->assign(temp3);
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
        
        while(i > 0 && (math::nd4j_abs<T>((*diagInterval)(i)) < almostZero || math::nd4j_abs<T>((*colVec0)(i)) < almostZero)) 
            --i;
        
        for(; i > 1; --i) {
            if( ((*diagInterval)(i) - (*diagInterval)(i-1)) < DataTypeUtils::eps<T>()*maxElem ) {
                if (math::nd4j_abs<T>((*diagInterval)(i) - (*diagInterval)(i-1)) >= epsBig)                     
                    throw "ops::helpers::SVD::deflation: diagonal elements are not properly sorted !";
                deflation2(col1, col1 + shift, row1W, col1W, i-1, i, len);
            }
        }
    }  

    delete colVec0;
    delete diagInterval;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
T SVD<T>::secularEq(const T diff, const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, const NDArray<T>& diagShifted, const T shift) {

  int len = permut.lengthOf();
  T res = 1.;
  for(int i=0; i<len; ++i) {

    int j = (int)permut(i);    
    res += col0(j)*col0(j) / ((diagShifted(j) - diff) * (diag(j) + shift + diff));
  }
  
  return res;

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVals(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, NDArray<T>& singVals, NDArray<T>& shifts, NDArray<T>& mus) {
  
    int len = col0.lengthOf();
    int curLen = len;
    
    while(curLen > 1 && col0(curLen-1) == (T)0.) 
        --curLen;

    for (int k = 0; k < len; ++k)  {
    
        if (col0(k) == (T)0. || curLen==1) {
    
            singVals(k) = k==0 ? col0(0) : diag(k);
            mus(k) = 0.;
            shifts(k) = k==0 ? col0(0) : diag(k);
            continue;
        } 
    
        T left = diag(k);
        T right;
    
        if(k==curLen-1)
            right = diag(curLen-1) + col0.template reduceNumber<simdOps::Norm2<T>>();
        else {
      
            int l = k+1;
            while(col0(l) == (T)0.) {
                ++l; 
                if(l >= curLen)
                    throw "ops::helpers::SVD::calcSingVals method: l >= curLen !";
            }
        
            right = diag(l);
        }
    
        T mid = left + (right - left) / (T)2.;
        T fMid = secularEq(mid, col0, diag, permut, diag, 0.);
        T shift = (k == curLen-1 || fMid > (T)0.) ? left : right;
        
        NDArray<T> diagShifted = diag - shift;
        
        T muPrev, muCur;
        if (shift == left) {
            muPrev = (right - left) * 0.1;
            if (k == curLen-1) 
                muCur = right - left;
            else               
                muCur = (right - left) * 0.5;
        }
        else {
            muPrev = -(right - left) * 0.1;
            muCur  = -(right - left) * 0.5;
        }

        T fPrev = secularEq(muPrev, col0, diag, permut, diagShifted, shift);
        T fCur = secularEq(muCur, col0, diag, permut, diagShifted, shift);
        
        if (math::nd4j_abs<T>(fPrev) < math::nd4j_abs<T>(fCur)) {      
            math::nd4j_swap<T>(fPrev, fCur);
            math::nd4j_swap<T>(muPrev, muCur);
        }
    
        bool useBisection = fPrev * fCur > (T)0.;
        while (fCur != (T).0 && 
               math::nd4j_abs<T>(muCur - muPrev) > (T)8. * DataTypeUtils::eps<T>() * math::nd4j_max<T>(math::nd4j_abs<T>(muCur), math::nd4j_abs<T>(muPrev)) 
               && math::nd4j_abs<T>(fCur - fPrev) > DataTypeUtils::eps<T>() && !useBisection) {
                        
            T a = (fCur - fPrev) / ((T)1./muCur - (T)1./muPrev);
            T b = fCur - a / muCur;      
            T muZero = -a/b;
            T fZero = secularEq(muZero, col0, diag, permut, diagShifted, shift);
      
            muPrev = muCur;
            fPrev = fCur;
            muCur = muZero;
            fCur = fZero;            
            
            if (shift == left  && (muCur < (T)0. || muCur > right - left)) 
                useBisection = true;
            if (shift == right && (muCur < -(right - left) || muCur > (T)0.)) 
                useBisection = true;
            if (math::nd4j_abs<T>(fCur) > math::nd4j_abs<T>(fPrev)) 
                useBisection = true;
        }
                    
        if (useBisection) {

            T leftShifted, rightShifted;
            if (shift == left) {
                leftShifted = DataTypeUtils::min<T>();
                rightShifted = (k==curLen-1) ? right : ((right - left) * (T)0.6); 
            }
            else {
           
                leftShifted = -(right - left) * (T)0.6;
                rightShifted = -DataTypeUtils::min<T>();
            }
      
            T fLeft  = secularEq(leftShifted,  col0, diag, permut, diagShifted, shift);
            T fRight = secularEq(rightShifted, col0, diag, permut, diagShifted, shift);            
            // if(fLeft * fRight >= (T)0.)
                // throw "ops::helpers::SVD::calcSingVals method: fLeft * fRight >= (T)0. !";        
      
            while (rightShifted - leftShifted > (T)2. * DataTypeUtils::eps<T>() * math::nd4j_max<T>(math::nd4j_abs<T>(leftShifted), math::nd4j_abs<T>(rightShifted))) {
            
                T midShifted = (leftShifted + rightShifted) / (T)2.;
                fMid = secularEq(midShifted, col0, diag, permut, diagShifted, shift);
                if (fLeft * fMid < (T)0.)        
                    rightShifted = midShifted;        
                else {
                    leftShifted = midShifted;
                    fLeft = fMid;
                }
            }
            muCur = (leftShifted + rightShifted) / (T)2.;
        }        
        singVals(k) = shift + muCur;
        shifts(k) = shift;
        mus(k) = muCur;        
    }

}


//////////////////////////////////////////////////////////////////////////
template <typename T> 
void SVD<T>::perturb(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, const NDArray<T>& singVals,  const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& zhat) {    
    
    int n = col0.lengthOf();
    int m = permut.lengthOf();
    if(m==0) {
        zhat.assign(0.);
        return;
    }
    
    int last = permut(m-1);
  
    for (int k = 0; k < n; ++k) {
        
        if (col0(k) == (T)0.)     
            zhat(k) = (T)0.;
        else {            
            T dk   = diag(k);
            T prod = (singVals(last) + dk) * (mus(last) + (shifts(last) - dk));

            for(int l = 0; l<m; ++l) {
                int i = permut(l);
                if(i!=k) {
                    int j = i<k ? i : (int)permut(l-1);
                    prod *= ((singVals(j)+dk) / ((diag(i)+dk))) * ((mus(j)+(shifts(j)-dk)) / ((diag(i)-dk)));
                }
            }
        T tmp = math::nd4j_sqrt<T>(prod);
        zhat(k) = col0(k) > (T)0. ? tmp : -tmp;
        }  
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVecs(const NDArray<T>& zhat, const NDArray<T>& diag, const NDArray<T>& perm, const NDArray<T>& singVals,
                             const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& U, NDArray<T>& V) {
  
    int n = zhat.lengthOf();
    int m = perm.lengthOf();
  
    for (int k = 0; k < n; ++k) {
        
        NDArray<T>* colU = U.subarray({{},{k, k+1}});
        *colU = 0.;
        NDArray<T>* colV = nullptr;
        
        if (_calcV) {
            colV = V.subarray({{},{k, k+1}});
            *colV = 0.;
        }

        if (zhat(k) == (T)0.) {
            (*colU)(k) = 1.;            
            
            if (_calcV)            
                (*colV)(k) = 1.;
        }
        else {
      
            for(int l = 0; l < m; ++l) {
                int i = (int)perm(l);
                U(i,k) = zhat(i)/(((diag(i) - shifts(k)) - mus(k)) )/( (diag(i) + singVals(k)));
            }
            U(n,k) = 0.;            
            *colU /= colU->template reduceNumber<simdOps::Norm2<T>>();            
    
            if (_calcV) {
        
                for(int l = 1; l < m; ++l){
                    int i = (int)perm(l);
                    V(i,k) = diag(i) * zhat(i) / (((diag(i) - shifts(k)) - mus(k)) )/( (diag(i) + singVals(k)));
                }
                V(0,k) = -1.;
                *colV /= colV->template reduceNumber<simdOps::Norm2<T>>();                
            }
        }
        delete colU;  
        if (_calcV)    
            delete colV;
    }
    
    NDArray<T>* colU = U.subarray({{},{n, n+1}});
    *colU = 0.;
    (*colU)(n) = 1.;
    delete colU;    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcBlockSVD(int col1, int size, NDArray<T>& U, NDArray<T>& singVals, NDArray<T>& V) {  
    
    const T almostZero = DataTypeUtils::min<T>();
    NDArray<T> col0 = _M({{col1, col1+size},{col1, col1+1}});
    NDArray<T>* diagP = _M({{col1, col1+size},{col1, col1+size}}).diagonal('c');
    NDArray<T> diag = *diagP;
    delete diagP;

    diag(0) = 0.;
    singVals = NDArray<T>(size, 1, _M.ordering(), _M.getWorkspace());
    U = NDArray<T>(size+1, size+1, _U.ordering(), _U.getWorkspace());
    if (_calcV) 
        V = NDArray<T>(size, size, _V.ordering(), _V.getWorkspace());
    
    int curSize = size;
    while(curSize > 1 && diag(curSize-1) == (T)0.) 
        --curSize;
    
    int m = 0; 
    std::vector<T> indices;
    for(int k = 0; k < curSize; ++k)
        if(math::nd4j_abs<T>(col0(k)) > almostZero)
            indices.push_back((T)k);            
  
    NDArray<T> permut(_M.ordering(), {1, (int)indices.size()}, indices, _M.getWorkspace());
    NDArray<T> shifts(size, 1, _M.ordering(), _M.getWorkspace());
    NDArray<T> mus(size, 1, _M.ordering(), _M.getWorkspace());
    NDArray<T> zhat(size, 1, _M.ordering(), _M.getWorkspace());
          
    calcSingVals(col0, diag, permut, singVals, shifts, mus);
    perturb(col0, diag, permut, singVals, shifts, mus, zhat);
    calcSingVecs(zhat, diag, permut, singVals, shifts, mus, U, V);      
        
    for(int i=0; i<curSize-1; ++i) {        
        
        if(singVals(i) > singVals(i+1)) {        
            math::nd4j_swap<T>(singVals(i),singVals(i+1));            
            NDArray<T>* temp1 = U.subarray({{},{i,i+1}});
            NDArray<T>* temp2 = U.subarray({{},{i+1,i+2}});                     
            NDArray<T> temp3 = *temp1;
            temp1->assign(temp2);
            temp2->assign(temp3);            
            delete temp1;
            delete temp2;
            
            if(_calcV) {
                NDArray<T>* temp1 = V.subarray({{},{i,i+1}});
                NDArray<T>* temp2 = V.subarray({{},{i+1,i+2}});                     
                NDArray<T> temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);            
                delete temp1;
                delete temp2;                    
            }
        }
    }
    
    NDArray<T>* temp1 = singVals.subarray({{0, curSize},{}});
    for (int e = 0; e < curSize / 2; ++e) {
        T tmp = (*temp1)(e);
        (*temp1)(e) = (*temp1)(curSize-1-e);
        (*temp1)(curSize-1-e) = tmp;
    }
    delete temp1;
    
    NDArray<T>* temp2 = U.subarray({{},{0, curSize}});
    for(int i = 0; i < curSize/2; ++i) {
        NDArray<T>* temp3 = temp2->subarray({{},{i,i+1}});
        NDArray<T>* temp4 = temp2->subarray({{},{curSize-1-i,curSize-i}});
        NDArray<T>  temp5 = *temp3;
        temp3->assign(temp4);
        temp4->assign(temp5);        
        delete temp3;
        delete temp4;
    }
    delete temp2;
    
    if (_calcV) {
        NDArray<T>* temp2 = V.subarray({{},{0, curSize}});
        for(int i = 0; i < curSize/2; ++i) {
            NDArray<T>* temp3 = temp2->subarray({{}, {i,i+1}});
            NDArray<T>* temp4 = temp2->subarray({{}, {curSize-1-i,curSize-i}});
            NDArray<T>  temp5 = *temp3;
            temp3->assign(temp4);
            temp4->assign(temp5);        
            delete temp3;
            delete temp4;
        }
        delete temp2;
    }     
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void SVD<T>::DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift) {
    
    // requires rows = cols + 1;
    const int n = col2 - col1 + 1;
    const int k = n/2;
    const T almostZero = DataTypeUtils::min<T>();
    T alphaK;
    T betaK; 
    T r0; 
    T lambda, phi, c0, s0;
    NDArray<T> l, f;
    

    if(n==2) { // if(n < 16)
            
        // for small matrices it is more efficient to use another algorithm Jacobi SVD
        return;
    }
      
    alphaK = _M(col1 + k, col1 + k);
    betaK  = _M(col1 + k + 1, col1 + k);
  
    DivideAndConquer(k + 1 + col1, col2, k + 1 + row1W, k + 1 + col1W, shift);
    DivideAndConquer(col1, k - 1 + col1, row1W, col1W + 1, shift + 1);

    if (_calcU) {
        lambda = _U(col1 + k, col1 + k);
        phi    = _U(col1 + k + 1, col2 + 1);
    } 
    else {
        lambda = _U(1, col1 + k);
        phi    = _U(0, col2 + 1);
    }
    
    r0 = math::nd4j_sqrt<T>((math::nd4j_abs<T>(alphaK * lambda) * math::nd4j_abs<T>(alphaK * lambda)) + math::nd4j_abs<T>(betaK * phi) * math::nd4j_abs<T>(betaK * phi));
    
    if (_calcU) {
        l = _U({{col1+k, col1+k+1},{col1, col1+k}});        
        f = _U({{col1+k+1, col1+k+2},{col1+k+1, col1+n}});        
    } 
    else {        
        l = _U({{1, 2},{col1, col1+k}});
        f = _U({{0, 1},{col1+k+1, col1+n}});
    }
    
    if (_calcV) 
        _V(row1W+k, col1W) = 1.;
  
    if (r0 < almostZero){
        c0 = 1.;
        s0 = 0.;
    }
    else {
        c0 = alphaK * lambda / r0;
        s0 = betaK * phi / r0;
    }
  
    if (_calcU) {
    
        NDArray<T> q1 = _U({{col1, col1+k+1},{col1+k, col1+k+1}});        
        
        for (int i = col1 + k - 1; i >= col1; --i) 
            _U({{col1, col1+k+1}, {i+1, i+2}}).assign(_U({{col1, col1+k+1}, {i, i+1}}));
    
        _U({{col1, col1+k+1}, {col1, col1+1}}).assign(q1 * c0);
        _U({{col1, col1+k+1}, {col2+1, col2+2}}).assign(q1 * (-s0));
        _U({{col1+k+1, col1+n+1}, {col1, col1+1}}).assign(_U({{col1+k+1, col1+n+1}, {col2+1, col2+2}}) * s0);
        _U({{col1+k+1, col1+n+1}, {col2+1, col2+2}}) *= c0;        
    } 
    else  {
    
        T q1 = _U(0, col1 + k);    
    
        for (int i = col1 + k - 1; i >= col1; --i) 
            _U(0, i+1) = _U(0, i);
    
        _U(0, col1) = q1 * c0;    
        _U(0, col2+1) = -q1*s0;    
        _U(1, col1) = _U(1, col2+1) * s0;     
        _U(1, col2 + 1) *= c0;
        _U({{1, 2}, {col1+1, col1+k+1}}) = 0.;
        _U({{0, 1}, {col1+k+1, col1+n}}) = 0.;    
    }
    
    _M(col1 + shift, col1 + shift) = r0;
    _M({{col1+shift+1, col1+shift+k+1}, {col1+shift, col1+shift+1}}).assign(l*alphaK);
    _M({{col1+shift+k+1, col1+shift+n}, {col1+shift, col1+shift+1}}).assign(f*betaK);    

    deflation(col1, col2, k, row1W, col1W, shift);
      
    NDArray<T> UofSVD, VofSVD, singVals;  
    calcBlockSVD(col1 + shift, n, UofSVD, singVals, VofSVD);
    
    if(_calcU)        
        _U({{col1, col1+n+1},{col1, col1+n+1}}).assign(mmul(_U({{col1, col1+n+1},{col1, col1+n+1}}), UofSVD));
    else 
        _U({{}, {col1, col1+n+1}}).assign(mmul(_U({{}, {col1, col1+n+1}}), UofSVD));
  
    if (_calcV)  
        _V({{row1W, row1W+n},{row1W, row1W+n}}).assign(mmul(_V({{row1W, row1W+n},{row1W, row1W+n}}), VofSVD));
        
    
    NDArray<T>* blockM = _M.subarray({{col1+shift, col1+shift+n},{col1+shift, col1+shift+n}});
    *blockM = 0.;
    NDArray<T>* diag = blockM->diagonal('c');        
    diag->assign(singVals);
    delete diag;
    delete blockM;

}




template class ND4J_EXPORT SVD<float>;
template class ND4J_EXPORT SVD<float16>;
template class ND4J_EXPORT SVD<double>;







}
}
}

