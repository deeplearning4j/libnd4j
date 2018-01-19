//
// Created by raver119 on 19.01.18.
//

#ifndef LIBND4J_S_T_B_H
#define LIBND4J_S_T_B_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    FORCEINLINE void _spaceToBatch(NDArray<T> *input, NDArray<T> *output, std::vector<int> &internal_input_shape, std::vector<int> &internal_output_shape, std::vector<int> &block_shape, std::vector<int> &paddings) {

    }
}
}
}

#endif //LIBND4J_S_T_B_H
