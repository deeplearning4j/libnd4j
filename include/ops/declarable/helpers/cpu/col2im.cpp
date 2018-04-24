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

                shape::printShapeInfoLinear("input shape", xShape);
                shape::printShapeInfoLinear("output shape", zShape);

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

                if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(guided)
                    for (int b = 0; b < samples; b++) {
                        T *input = dx + (b * strideex);
                        T *result = dz + (b * outStride[0]);

                        for (int channel = depth; channel--; result += outStride[1]) {
                            for (int kernel_row = 0; kernel_row < kY; kernel_row++) {
                                for (int kernel_col = 0; kernel_col < kX; kernel_col++) {
                                    int input_row = -pY + kernel_row * dY;
                                    for (int output_rows = height_col; output_rows; output_rows--) {
                                        if (!is_a_ge_zero_and_a_lt_b(input_row, imgY)) {
                                            input += width_col * stridecol;
                                        } else {
                                            int input_col = -pX + kernel_col * dX;

                                            if (channel == depth && is_a_ge_zero_and_a_lt_b(input_col, imgX))
                                                result[input_row * outStride[2] + input_col * outStride[3]] = (T) 0.0f;

                                            for (int output_col = width_col; output_col; output_col--) {
                                                if (is_a_ge_zero_and_a_lt_b(input_col, imgX)) {
                                                    result[input_row * outStride[2] + input_col * outStride[3]] += *input;
                                                    //result[input_row * imgX + input_col] += *input;
                                                }
                                                input += stridecol;
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
#pragma omp parallel for schedule(guided) proc_bind(close)
                    for (int i = 0; i < n; i++) {
                        T val = 0;
                        int w_im = i % imgX + pX;
                        int h_im = (i / imgX) % imgY + pY;
                        int c_im = i / (imgX * imgY);

                        int num_im = c_im / depth;
                        int depth_im = c_im % depth;

                        // compute the start and end of the output
                        // These are the indexes for dimensions ??? in the 6d col matrix
                        int w_col_start = (w_im < kEffectiveW) ? 0 : (w_im - kEffectiveW) / sX + 1;
                        int w_col_end = nd4j::math::nd4j_min<int>(w_im / sX + 1, width_col);

                        int h_col_start = (h_im < kEffectiveH) ? 0 : (h_im - kEffectiveH) / sY + 1;
                        int h_col_end = nd4j::math::nd4j_min<int>(h_im / sY + 1, height_col);


                        //Iterate over col entries in the 6d array... these are added up
                        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
                            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                                int h_k = (h_im - h_col * sY);
                                int w_k = (w_im - w_col * sX);

                                if(h_k % dY == 0 && w_k % dX == 0){
                                    h_k /= dY;
                                    w_k /= dX;

                                    int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;
                                    val += dx[data_col_index];
                                }
                            }
                        }
                        int i_f = 0;
                        int i_c = i;
                        for (int dim = 3; dim >= 0; dim--)
                        {
                            i_f += (i_c % outShape[dim])  * outStride[dim];
                            i_c = i_c / outShape[dim];
                        }
                        dz[i_f] += val;
                    }
                }
            };

            template void _col2im<float>(nd4j::graph::LaunchContext& context, float *dx, float *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
            template void _col2im<float16>(nd4j::graph::LaunchContext& context, float16 *dx, float16 *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
            template void _col2im<double>(nd4j::graph::LaunchContext& context, double *dx, double *result, int *zShape, int *xShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
        }
    }
}
