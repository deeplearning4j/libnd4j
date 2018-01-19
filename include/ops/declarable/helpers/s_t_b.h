//
// Created by raver119 on 19.01.18.
//

#ifndef LIBND4J_S_T_B_H
#define LIBND4J_S_T_B_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {
    // this method MUST be platform-specific
    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(T *ptrSpace, const int *space_shape, const int *space_strides, const int *block_shape, const int *pad_start, const int *block_offsets, T *ptrBatch, const int *batch_shape, const int *batch_strides);


    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    FORCEINLINE void _prepare(NDArray<T> * space, NDArray<T> *batch, int block_array[NUM_BLOCK_DIMS], int padding_array[NUM_BLOCK_DIMS]) {

        int pad_start[NUM_BLOCK_DIMS];
        int block_shape[NUM_BLOCK_DIMS];
        int space_shape[NUM_BLOCK_DIMS];
        int batch_shape[NUM_BLOCK_DIMS];

        const int batch_size = batch->sizeAt(0);
        const int space_size = space->sizeAt(0);

#pragma unroll
        for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; block_dim++) {
            pad_start[block_dim] = padding_array[block_dim * 2];
            block_shape[block_dim] = block_array[block_dim];
            space_shape[block_dim] = space->sizeAt(block_dim + 1);
            batch_shape[block_dim] = batch->sizeAt(block_dim + 1);
        }

        int space_strides[NUM_BLOCK_DIMS + 2];
        int batch_strides[NUM_BLOCK_DIMS + 2];
        space_strides[NUM_BLOCK_DIMS + 1] = 1;
        batch_strides[NUM_BLOCK_DIMS + 1] = 1;


        // TODO: this loop should be moved to _execute phase
        for (int batch_b = 0; batch_b < batch_size; batch_b++) {
            const int space_b = batch_b % space_size;
            int block_index = batch_b / space_size;
            int block_offsets[NUM_BLOCK_DIMS];
            for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; block_dim--) {
                block_offsets[block_dim] = block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
                block_index /= block_shape[block_dim];
            }


            // exec
            _execute(space->buffer() + space_b * space_strides[0], space_shape, &space_strides[1], block_shape, pad_start, block_offsets, batch_shape, &batch_strides[1], batch->buffer() + batch_b * batch_strides[0]);
        }
    };


    template <typename T>
    FORCEINLINE void _spaceToBatch(NDArray<T> *input, NDArray<T> *output, std::vector<int> &internal_input_shape, std::vector<int> &internal_output_shape, std::vector<int> &block_shape, std::vector<int> &paddings) {

    }
}
}
}

#endif //LIBND4J_S_T_B_H
