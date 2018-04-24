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

                int n = samples * depth * height_col * width_col;

                int hw = height * width;

                int _width = width * stridew;

                //if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(guided)
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
                                            for (int output_col = width_col; output_col; output_col--) {
                                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                                    *(result++) = input[input_row * strideh + input_col * stridew];
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

                /*} else {
#pragma omp parallel for schedule(guided) proc_bind(close)
                    for (int index = 0; index < n; index++) {
                        int h_index = index / width_col;
                        int h_col = h_index % height_col;
                        int w_col = index % width_col;

                        int c_im = h_index / height_col;
                        int c_col = c_im * kSize;

                        int depth_im = c_im % depth;
                        int num_im = c_im / depth;
                        int h_offset = h_col * sY - pY;
                        int w_offset = w_col * sX - pX;

                        T* data_col_ptr = dz;

                        int i_c = (c_col * height_col + h_col) * width_col + w_col;
                        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                        T* data_im_ptr = dx;

                        data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

                        for (int i = 0; i < kY; ++i) {
                            for (int j = 0; j < kX; ++j) {
                                int h_im = h_offset + i * dY;
                                int w_im = w_offset + j * dX;
                                int i_f = 0;
                                int i_c_temp = i_c;
                                for (int dim = 5; dim >= 0; dim--) {
                                    i_f += (i_c_temp % outShape[dim])  * outStride[dim];
                                    i_c_temp = i_c_temp / outShape[dim];
                                }
                                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width){
                                    dz[i_f] = data_im_ptr[i * dY * strideh + j * dX * stridew];
                                } else dz[i_f] = zeroPadVal;

                                //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                                data_col_ptr += height_col * width_col;
                                i_c += height_col * width_col;
                            }
                        }
                    }
                }
                */
            }


            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *result, float *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *result, float16 *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *result, double *dx, int *zShape, int *xShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, double zeroPadVal);
        }
    }
}