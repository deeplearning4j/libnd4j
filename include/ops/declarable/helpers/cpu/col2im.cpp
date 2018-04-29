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

            // input [bS, iC, kH, kW, oH, oW] is de-convoluted to output [bS, iC, iH, iW]
            template <typename T>
            void _col2im(nd4j::graph::LaunchContext& context, T *out, T *in, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW) {

                int *inShape = shape::shapeOf(inShapeInfo);
                int *inStride = shape::stride(inShapeInfo);

                int kH = inShape[2];
                int kW = inShape[3];

                int *outShape = shape::shapeOf(outShapeInfo);
                int *outStride = shape::stride(outShapeInfo);

                int bS = outShape[0];
                int iC = outShape[1];

                int oH = inShape[4];                            // (iH + 2 * pH- kH) / sH + 1;
                int oW = inShape[5];                            // (iW + 2 * pW- kW) / sW + 1;

                const int inStepOW = oW * inStride[5];

                if (shape::order(inShapeInfo) == 'c' &&  shape::order(outShapeInfo) == 'c' && shape::strideDescendingCAscendingF(inShapeInfo) && shape::strideDescendingCAscendingF(outShapeInfo)) {

#pragma omp parallel for schedule(guided)
                    for (int b = 0; b < bS; b++) {
                        T *input = in + (b * inStride[0]);
                        T *output = out + (b * outStride[0]);


                        for (int channel = 0; channel < iC; ++channel, output += outStride[1]) {

                            for (int kRow = 0; kRow < kH; ++kRow) {                                
                                int inRowStart = -pH + kRow * dH;
                                
                                for (int kCol = 0; kCol < kW; ++kCol) {
                                    int inRow = inRowStart;
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH) {
                                        
                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            input += inStepOW;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * outStride[2];

                                            if (channel == iC && is_a_ge_zero_and_a_lt_b(inCol, iW))
                                                output[h + inCol * outStride[3]] = (T) 0.0f;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW, input += inStride[5]) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) {
                                                    T& result = output[h + inCol * outStride[3]];
                                                    result += *input;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } 
                else {
#pragma omp parallel for schedule(guided) 
                    for (int b = 0; b < bS; b++) {                        
                        T* output = out + (b * outStride[0]);
                        T* in0 = in + b * inStride[0];

                        for (int channel = 0; channel < iC; ++channel, output+=outStride[1], in0+=inStride[1]) {
                            T* in1 = in0;

                            for (int kRow = 0; kRow < kH; ++kRow, in1+=inStride[2]) {  
                                T* in2 = in1;
                                int inRowStart = -pH + kRow * dH;
                                
                                for (int kCol = 0; kCol < kW; ++kCol, in2+=inStride[3]) {
                                    T* in3 = in2;
                                    int inRow = inRowStart;
                                    int inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow+=sH, in3+=inStride[4]) {                                        
                                        T* in4 = in3;

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            in4 += inStepOW;
                                        } 
                                        else {
                                            int inCol = inColStart;
                                            Nd4jIndex h = inRow * outStride[2];                                             

                                            if (channel == iC && is_a_ge_zero_and_a_lt_b(inCol, iW))
                                                output[h + inCol * outStride[3]] = (T) 0.0f;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol+=sW, in4+=inStride[5]) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) {
                                                    T& result = output[h + inCol * outStride[3]];
                                                    result += *in4;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };

            template void _col2im<float>(nd4j::graph::LaunchContext& context, float *in, float *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<float16>(nd4j::graph::LaunchContext& context, float16 *in, float16 *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<double>(nd4j::graph::LaunchContext& context, double *in, double *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
        }
    }
}
