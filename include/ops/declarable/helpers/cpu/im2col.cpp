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

            // input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]
            template <typename T>
            void _im2col(nd4j::graph::LaunchContext& context, T *out, T *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, T zeroPadVal) {
                int kSize = kH * kW;

                int *outShape  = shape::shapeOf(zShape);
                char outOrder  = shape::order(zShape);
                int *outStride = shape::stride(zShape);

                int *inShape = shape::shapeOf(xShape);
                int *inStride = shape::stride(xShape);

                int bS = inShape[0];
                int iC = inShape[1];
                int iH = inShape[2];
                int iW = inShape[3];

                int oH = outShape[4];
                int oW = outShape[5];

                if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(static) proc_bind(close)
                    for (int b = 0; b < bS; b++) {
                        T *input = in + (b * inStride[0]);
                        T *output = out + (b * outStride[0]);                        

                        for (int channel = 0; channel < iC; ++channel, input += inStride[1]) { 

                            for (int kRow = 0; kRow < kH; kRow++) {
                                int inRowStart = -pH + kRow * dH; 

                                for (int kCol = 0; kCol < kW; kCol++) {                                    
                                    int inRow = inRowStart;                                    
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH) {                                        

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) 
                                            for (int outCol = 0; outCol < oW; ++outCol, ++output) {
                                                *output = zeroPadVal;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * inStride[2];

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW, ++output) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    *output = input[h + inCol * inStride[3]];
                                                else 
                                                    *output = zeroPadVal;
                                            }
                                        }                                        
                                    }
                                }
                            }
                        }
                    }
                } 
                else {
                                    
#pragma omp parallel for schedule(static) proc_bind(close)
                    for (int b = 0; b < bS; b++) {
                        T *input = in + (b * inStride[0]);
                        T* out0  = out + b * outStride[0];

                        for (int channel = 0; channel < iC; ++channel, input += inStride[1], out0+=outStride[1]) {
                            T* out1 = out0;

                            for (int kRow = 0; kRow < kH; kRow++, out1 += outStride[2]) {
                                T* out2 = out1;
                                int inRowStart = -pH + kRow * dH; 

                                for (int kCol = 0; kCol < kW; kCol++, out2 += outStride[3]) {
                                    T* out3 = out2;
                                    int inRow = inRowStart;                                    
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH, out3 += outStride[4]) {
                                        T* out4 = out3;

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) 
                                            for (int outCol = 0; outCol < oW; ++outCol, out4 += outStride[5]) {
                                                *out4 = zeroPadVal;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * inStride[2];

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW, out4 += outStride[5]) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    *out4 = input[h + inCol * inStride[3]];
                                                else 
                                                    *out4 = zeroPadVal;
                                            }
                                        }                                        
                                    }
                                }
                            }
                        }
                    }

                }
            }

            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *output, float *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *output, float16 *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *output, double *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, double zeroPadVal);
        }
    }
}