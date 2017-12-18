//
// Created by Yurii Shyrma on 18.12.2017.
//

#ifndef LIBND4J_HOUSEHOLDER_H
#define LIBND4J_HOUSEHOLDER_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


	
    template <typename T>
    void houseHolderForVector(const NDArray<T>& inVector, NDArray<T>& outVector, T& squaredNorm, T& coeff);

    
    

}
}
}


#endif //LIBND4J_HOUSEHOLDER_H
