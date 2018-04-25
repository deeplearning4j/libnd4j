//
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/im2col.h>
#include <atomic>

namespace nd4j {
    namespace ops {
        namespace helpers {

            FORCEINLINE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
                return static_cast<unsigned>(a) < static_cast<unsigned>(b);
            }


            FORCEINLINE int zindex(int &index, int length, int *outShape, int *outStride) {
                int denom = length;
                int offset = 0;
                int idx = index;

#pragma unroll
                for(int i = 0; i < 6; i++) {
                    int v = 0;
                    denom /= outShape[i];
                    if (denom > 0) {
                        v = idx / denom;
                        idx %= denom;
                    }

                    if (outShape[i] > 1) {
                        offset += v * outStride[i];
                    }
                }

                ++index;

                return offset;
            }

            template <typename T>
            void _im2col(nd4j::graph::LaunchContext& context, T *dz, T *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, T zeroPadVal) {
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

                if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(static) proc_bind(close)
                    for (int b = 0; b < samples; b++) {
                        T *input = dx + (b * strideex);
                        T *result = dz + (b * outStride[0]);

                        for (int channel = depth; channel--; input += stridech) {
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
                                            Nd4jIndex _h = input_row * strideh;

                                            for (int output_col = width_col; output_col; output_col--) {
                                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                                    *(result++) = input[_h + input_col * stridew];
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
                } else {

                        int l = shape::prod(outShape, 6);

#pragma omp parallel for schedule(static) proc_bind(close)
                        for (int b = 0; b < samples; b++) {
                            T *input = dx + (b * strideex);
                            T *result = dz;// + (b * outStride[0]);
                            int index = b * (outShape[1] * outShape[2] * outShape[3] * outShape[4] * outShape[5]);

                            for (int channel = depth; channel--; input += stridech) {
                                for (int kernel_row = 0; kernel_row < kY; kernel_row++) {
                                    for (int kernel_col = 0; kernel_col < kX; kernel_col++) {
                                        int input_row = -pY + kernel_row * dY;
                                        for (int output_rows = height_col; output_rows; output_rows--) {
                                            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                                                for (int output_cols = width_col; output_cols; output_cols--) {
                                                    result[zindex(index, l, outShape, outStride)] = zeroPadVal;
                                                }
                                            } else {
                                                int input_col = -pX + kernel_col * dX;
                                                Nd4jIndex _h = input_row * strideh;

                                                for (int output_col = width_col; output_col; output_col--) {
                                                    if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                                        result[zindex(index, l, outShape, outStride)] = input[_h + input_col * stridew];
                                                    } else {
                                                        result[zindex(index, l, outShape, outStride)] = zeroPadVal;
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
                    }
            }


            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *result, float *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *result, float16 *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *result, double *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
        }
    }
}