//
// Created by raver119 on 19.01.18.
//

#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
namespace helpers {


    template <int N, bool B2S>
    struct SpaceToBatchHelper {
        template <typename T>
        static void run(T *ptrSpace, const int *space_shape, const int *space_strides, const int *block_shape, const int *pad_start, const int *block_offsets, T *ptrBatch, const int *batch_shape, const int *batch_strides) {
            for (int batch_pos = 0; batch_pos < batch_shape[0]; batch_pos++) {
                const int space_pos = batch_pos * block_shape[0] + block_offsets[0] - pad_start[0];
                if (space_pos >= 0 && space_pos <= space_shape[0]) {
                    SpaceToBatchHelper<N - 1, B2S>::run(ptrSpace + space_pos * space_strides[0], space_shape + 1, space_strides + 1, block_shape + 1, pad_start + 1, block_offsets + 1, ptrBatch, batch_shape + 1, batch_strides + 1);
                } else {
                    if (!B2S)
                        for (int i = 0; i < batch_strides[0]; i++)
                            ptrBatch[i] = (T) 0.f;
                }

                ptrBatch += batch_strides[0];
            }
        }
    };

    template <bool B2S>
    struct SpaceToBatchHelper<0, B2S> {
        template <typename T>
        static void run(T *ptrSpace, const int *space_shape, const int *space_strides, const int *block_shape, const int *pad_start, const int *block_offsets, T *ptrBatch, const int *batch_shape, const int *batch_strides) {
            for (int i = 0; i < batch_strides[-1]; i++)
                if (B2S)
                    ptrSpace[i] = ptrBatch[i];
                else
                    ptrBatch[i] = ptrSpace[i];
        }
    };

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(T *ptrSpace, const int *space_shape, const int *space_strides, const int *block_shape, const int *pad_start, const int *block_offsets, T *ptrBatch, const int *batch_shape, const int *batch_strides) {
        SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace, space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };
}
}
}