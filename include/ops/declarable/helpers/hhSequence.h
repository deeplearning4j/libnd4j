//
// Created by Yurii Shyrma on 02.01.2018
//

#ifndef LIBND4J_HHSEQUENCE_H
#define LIBND4J_HHSEQUENCE_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class HHsequence {

    public:
    
    /*
    *  matrix containing the Householder vectors
    */
    NDArray<T> _vectors;        

    /*
    *  vector containing the Householder coefficients
    */
    NDArray<T> _coeffs;    
    
    /*
    *  shift of the Householder sequence 
    */
    int _length;

    /*
    *  length of the Householder sequence
    */
    int _shift;        

    /*
    *  type of sequence, type = 'u' (acting on columns) or type = 'v' (acting on rows)
    */
    int _type;        

    /*
    *  constructor
    */
    HHsequence(const NDArray<T>& vectors, const NDArray<T>& coeffs, const char type);

    /**
    *  this method mathematically multiplies input matrix on Householder sequence from the left H0*H1*...Hn * matrix
    * 
    *  matrix - input matrix to be multiplied
    */                       
    void mulLeft(NDArray<T>& matrix) const;

    



};

    



}
}
}


#endif //LIBND4J_HHSEQUENCE_H
