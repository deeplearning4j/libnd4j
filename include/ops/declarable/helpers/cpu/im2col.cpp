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
            void _im2col(nd4j::graph::LaunchContext& context, T *out, T *in, Nd4jLong *zShape, Nd4jLong *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, T zeroPadVal) {

                const int *outShape  = shape::shapeOf(zShape);
                const char outOrder  = shape::order(zShape);
                const int *outStride = shape::stride(zShape);
                const int *inShape = shape::shapeOf(xShape);
                const int *inStride = shape::stride(xShape);

                const int bS = inShape[0];
                const int iC = inShape[1];
                const int iH = inShape[2];
                const int iW = inShape[3];
                const int oH = outShape[4];
                const int oW = outShape[5];
                const int outStride0  = outStride[0];
                const int outStride1  = outStride[1];
                const int outStride2  = outStride[2];
                const int outStride3  = outStride[3];
                const int outStride4  = outStride[4];
                const int outStride5  = outStride[5];
                const int inStride0   = inStride[0];
                const int inStride1   = inStride[1];
                const int inStride2   = inStride[2];
                const int inStride3   = inStride[3];

                const T* in0End = in + inStride1 * iC;
                const int kRowEnd = -pH + kH * dH;
                const int kColEnd = -pW + kW * dW;
                const int oHW = oH * oW;
                const int inRowEnd = oH * sH;
                const int inColEnd = oW * sW;
                                
                T *in0, *in1;

                if (shape::order(xShape) == 'c' &&  shape::order(zShape) == 'c' && shape::strideDescendingCAscendingF(xShape) && shape::strideDescendingCAscendingF(zShape)) {

#pragma omp parallel for schedule(static) proc_bind(close) private(in0, in1)
                    for (int b = 0; b < bS; b++) {
                        T *output = out + (b * outStride0);                        

                        for (in0 = in + (b * inStride0); in0 < (b * inStride0) + in0End; in0 += inStride1) {

                            for (int kRow = -pH; kRow < kRowEnd; kRow+=dH) {

                                for (int kCol = -pW; kCol < kColEnd; kCol+=dW) {

                                    for (int inRow = kRow; inRow < kRow + inRowEnd; inRow+=sH) {

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            for (T* outStart = output; output < outStart + oW; ++output) 
                                                *output = zeroPadVal;
                                        } 
                                        else {                                            
                                            in1 = in0 + inRow * inStride2;

                                            for (int inCol = kCol; inCol < kCol + inColEnd; inCol+=sW, ++output)
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    *output = *(in1 + inCol * inStride3);                                                
                                                else 
                                                    *output = zeroPadVal;                                            
                                        }        
                                    }
                                }
                            }
                        }
                    }
                } 
                else {
                    const int col4End = oW * outStride5;
                    T *out0, *out1, *out2, *out3, *out4;
#pragma omp parallel for private(in0, in1, out0, out1, out2, out3, out4)
                    for (int b = 0; b < bS; b++) {                        
                        out0  = out + b * outStride0;

                        for (in0 = in + (b * inStride0); in0 < (b * inStride0) + in0End; in0+=inStride1, out0+=outStride1) {                        
                            out1 = out0;

                            for (int kRow = -pH; kRow < kRowEnd; kRow+=dH, out1+=outStride2) {
                                out2 = out1;

                                for (int kCol = -pW; kCol < kColEnd; kCol+=dW, out2+=outStride3) {
                                    out3 = out2;                                    

                                    for (int inRow = kRow; inRow < kRow + inRowEnd; inRow+=sH, out3+=outStride4) {
                                        out4 = out3;

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            for (; out4 < out3 + col4End; out4+=outStride5) 
                                               *out4 = 0.;                                                                             
                                        } 
                                        else {                                            
                                            in1 = in0 +  inRow * inStride2;

                                            for (int inCol = kCol; inCol < kCol + inColEnd; inCol+=sW, out4+=outStride5)
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) 
                                                    *out4 = *(in1 + inCol * inStride3);
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

            template void _im2col<float>(nd4j::graph::LaunchContext& context, float *output, float *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float zeroPadVal);
            template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *output, float16 *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float16 zeroPadVal);
            template void _im2col<double>(nd4j::graph::LaunchContext& context, double *output, double *in, int *zShape, int *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, double zeroPadVal);
        }
    }
}