//
// Created by Yurii Shyrma on 03.01.2018
//

#ifndef LIBND4J_SVD_H
#define LIBND4J_SVD_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hhSequence.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class SVD {

    public:
    
    int _switchSize = 10;

    NDArray<T> _M;
    NDArray<T> _S;
    NDArray<T> _U;
    NDArray<T> _V;
    
    int _diagSize;

    bool _transp;
    bool _calcU;
    bool _calcV;
    bool _fullUV;

    /**
    *  constructor
    */
    SVD(const NDArray<T>& matrix, const int switchSize, const bool calcV, const bool calcU, const bool fullUV);

    SVD(const NDArray<T>& matrix, const int switchSize, const bool calcV, const bool calcU, const bool fullUV, const char t);

    void deflation1(int col1, int shift, int ind, int size);
    
    void deflation2(int col1U , int col1M, int row1W, int col1W, int ind1, int ind2, int size);
    
    void deflation(int col1, int col2, int ind, int row1W, int col1W, int shift);    
    
    T secularEq(const T diff, const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T> &permut, const NDArray<T>& diagShifted, const T shift);

    void calcSingVals(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, NDArray<T>& singVals, NDArray<T>& shifts, NDArray<T>& mus);

    void perturb(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, const NDArray<T>& singVals,  const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& zhat);

    void calcSingVecs(const NDArray<T>& zhat, const NDArray<T>& diag, const NDArray<T>& perm, const NDArray<T>& singVals, const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& U, NDArray<T>& V);

    void calcBlockSVD(int firstCol, int size, NDArray<T>& U, NDArray<T>& singVals, NDArray<T>& V);

    void DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift);

    void exchangeUV(const HHsequence<T>& hhU, const HHsequence<T>& hhV, const NDArray<T> U, const NDArray<T> V);

    void evalData(const NDArray<T>& matrix);

};

    



}
}
}


#endif //LIBND4J_SVD_H
