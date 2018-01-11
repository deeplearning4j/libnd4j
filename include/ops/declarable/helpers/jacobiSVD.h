//
// Created by Yurii Shyrma on 11.01.2018
//

#ifndef LIBND4J_JACOBISVD_H
#define LIBND4J_JACOBISVD_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hhSequence.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class JacobiSVD {

    public:        
    
        NDArray<T> _M;
        NDArray<T> _S;
        NDArray<T> _U;
        NDArray<T> _V;
    
        int _diagSize;
        int _rows;
        int _cols;

        // bool _transp;
        bool _calcU;
        bool _calcV;
        bool _fullUV;

};



}
}
}


#endif //LIBND4J_JACOBISVD_H
