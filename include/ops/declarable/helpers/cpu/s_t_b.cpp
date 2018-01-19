//
// Created by raver119 on 19.01.18.
//

#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(T *ptrSpace, const int *space_shape, const int *space_strides, const int *block_shape, const int *pad_start, const int *block_offsets, T *ptrBatch, const int *batch_shape, const int *batch_strides) {
        for (int batch_pos = 0; batch_pos < batch_shape[0]; batch_pos++) {

        }
    };
}
}
}