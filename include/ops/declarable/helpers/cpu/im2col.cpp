//
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/im2col.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            FORCEINLINE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
                return static_cast<unsigned>(a) < static_cast<unsigned>(b);
            }

            template <typename T>
            void _im2col(nd4j::graph::LaunchContext& context, T *result, T *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, T zeroPadVal) {
                int kSize = kY * kX;

                int *outShape = shape::shapeOf(zShape);
                char resultOrder = shape::order(zShape);
                int *outStride = shape::stride(zShape);

                int *inShape = shape::shapeOf(xShape);
                int *inStride = shape::stride(xShape);

                int samples = inShape[0];
                int depth = inShape[1];
                int height = inShape[2];
                int width = inShape[3];


                int strideex = inStride[0];
                int stridech = inStride[1];
                int strideh = inStride[2];
                int stridew = inStride[3];

                int height_col = outShape[4];
                int width_col = outShape[5];

                int n = samples * depth * height_col * width_col;

                for (int channel = depth; channel--; dx += stridech) {
                    for (int kernel_row = 0; kernel_row < kY; kernel_row++) {
                        for (int kernel_col = 0; kernel_col < kX; kernel_col++) {
                            int input_row = -pY + kernel_row * dY;
                            for (int output_rows = height_col; output_rows; output_rows--) {
                                if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                                    for (int output_cols = width_col; output_cols; output_cols--) {
                                        *(result++) = zeroPadVal;
                                    }
                                } else {
                                    int input_col = -pX + kernel_col * dX;
                                    for (int output_col = width_col; output_col; output_col--) {
                                        if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                            *(result++) = dx[input_row * width + input_col];
                                        } else {
                                            *(result++) = zeroPadVal;
                                        }
                                        input_col += sX;
                                    }
                                }
                                input_row += sY;
                            }
                        }
                    }
                }
            }


            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *result, float *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *result, float16 *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *result, double *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
        }
    }
}