//
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            FORCEINLINE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
                return static_cast<unsigned>(a) < static_cast<unsigned>(b);
            }

            template <typename T>
            void _col2im(nd4j::graph::LaunchContext& context, T *dz, T *dx, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX) {
                int *inShape = shape::shapeOf(xShape);
                int *inStride = shape::stride(xShape);

                int strideex = inStride[0];
                int stridech = inStride[1];
                int stridekrow = inStride[2];
                int stridekcol = inStride[3];
                int striderow = inStride[4];
                int stridecol = inStride[5];

                int kY = inShape[2];
                int kX = inShape[3];

                int *outShape = shape::shapeOf(zShape);
                int *outStride = shape::stride(zShape);

                int samples = outShape[0];
                int depth = outShape[1];

                int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
                int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

                int n = samples * depth * imgY * imgX;

                //Effective kernel size, accounting for dilation
                int kEffectiveW = kX + (kX - 1) * (dX - 1);
                int kEffectiveH = kY + (kY - 1) * (dY - 1);

#pragma omp parallel for schedule(guided)
                for (int b = 0; b < samples; b++) {
                    T *input = dx + (b * strideex);
                    T *result = dz + (b * outStride[0]);

                    for (int channel = depth; channel--; result += depth) {
                        for (int kernel_row = 0; kernel_row < kY; kernel_row++) {
                            for (int kernel_col = 0; kernel_col < kX; kernel_col++) {
                                int input_row = -pY + kernel_row * dY;
                                for (int output_rows = height_col; output_rows; output_rows--) {
                                    if (!is_a_ge_zero_and_a_lt_b(input_row, imgY)) {
                                        input += width_col;
                                    } else {
                                        int input_col = -pX + kernel_col * dX;
                                        for (int output_col = width_col; output_col; output_col--) {
                                            if (is_a_ge_zero_and_a_lt_b(input_col, imgX)) {
                                                result[input_row * imgX + input_col] += *input;
                                            }
                                            input++;
                                            input_col += sX;
                                        }
                                    }
                                    input_row += sY;
                                }
                            }
                        }
                    }
                }
            };

            template void _col2im<float>(nd4j::graph::LaunchContext& context, float *dx, float *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
            template void _col2im<float16>(nd4j::graph::LaunchContext& context, float16 *dx, float16 *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
            template void _col2im<double>(nd4j::graph::LaunchContext& context, double *dx, double *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
        }
    }
}
