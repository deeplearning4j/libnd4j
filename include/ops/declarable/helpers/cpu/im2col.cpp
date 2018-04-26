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


            FORCEINLINE int zindex(int &outIndex, int lenOut, int *outShape, int *outStride) {
                int denom = lenOut;
                int offset = 0;
                int idx = outIndex;

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

                ++outIndex;

                return offset;
            }

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


                int strideBS = inStride[0];
                int strideIC = inStride[1];
                int strideH = inStride[2];
                int strideW = inStride[3];

                int oH = outShape[4];
                int oW = outShape[5];

                if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(static) proc_bind(close)
                    // for (int b = 0; b < bS; b++) {
                    //     T *input = in + (b * strideBS);
                    //     T *result = out + (b * outStride[0]);

                    //     for (int channel = iC; channel--; input += strideIC) {
                    //         for (int kRow = 0; kRow < kH; kRow++) {
                    //             for (int kCol = 0; kCol < kW; kCol++) {
                    //                 int inRow = -pH + kRow * dH;
                    //                 for (int outRow = oH; outRow; outRow--) {
                    //                     if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                    //                         for (int outCol = oW; outCol; outCol--) {
                    //                             *(result++) = zeroPadVal;
                    //                         }
                    //                     } else {
                    //                         int inCol = -pW + kCol * dW;
                    //                         Nd4jIndex h = inRow * strideH;

                    //                         for (int outCol = oW; outCol; outCol--) {
                    //                             if (is_a_ge_zero_and_a_lt_b(inCol, iW)) {
                    //                                 *(result++) = input[h + inCol * strideW];
                    //                             } else {
                    //                                 *(result++) = zeroPadVal;
                    //                             }
                    //                             inCol += sW;
                    //                         }
                    //                     }
                    //                     inRow += sH;
                    //                 }
                    //             }
                    //         }
                    //     }
                    // }
                    for (int b = 0; b < bS; b++) {
                        T *input = in + (b * strideBS);
                        T *result = out + (b * outStride[0]);                        

                        for (int channel = 0; channel < iC; ++channel, input += strideIC) { 

                            for (int kRow = 0; kRow < kH; kRow++) {
                                int inRowStart = -pH + kRow * dH; 

                                for (int kCol = 0; kCol < kW; kCol++) {                                    
                                    int inRow = inRowStart;                                    
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH) {                                        

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) 
                                            for (int outCol = 0; outCol < oW; ++outCol) {
                                                *(result++) = zeroPadVal;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * strideH;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    *(result++) = input[h + inCol * strideW];
                                                else 
                                                    *(result++) = zeroPadVal;
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
                        T *input = in + (b * strideBS);
                        int outIndex0 = b * outStride[0];

                        for (int channel = 0; channel < iC; ++channel, input += strideIC, outIndex0+=outStride[1]) {
                            int outIndex1 = outIndex0;

                            for (int kRow = 0; kRow < kH; kRow++, outIndex1 += outStride[2]) {
                                int outIndex2 = outIndex1;                                
                                int inRowStart = -pH + kRow * dH; 

                                for (int kCol = 0; kCol < kW; kCol++, outIndex2 += outStride[3]) {
                                    int outIndex3 = outIndex2;
                                    int inRow = inRowStart;                                    
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH, outIndex3 += outStride[4]) {
                                        int outIndex4 = outIndex3;

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) 
                                            for (int outCol = 0; outCol < oW; ++outCol, outIndex4 += outStride[5]) {
                                                out[outIndex4] = zeroPadVal;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * strideH;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW, outIndex4 += outStride[5]) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    out[outIndex4] = input[h + inCol * strideW];
                                                else 
                                                    out[outIndex4] = zeroPadVal;
                                            }
                                        }                                        
                                    }
                                }
                            }
                        }
                    }

                }
            }

            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *result, float *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *result, float16 *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *result, double *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, double zeroPadVal);
        }
    }
}