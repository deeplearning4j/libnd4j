//
// Created by raver119 on 07.10.2017.
//

#include <gemm.h>
#include <cstdio>

namespace nd4j {
    namespace blas {

        template <typename T>
        int GEMM<T>::linearIndexC(int rows, int cols, int r, int c) {
            return (r * cols + c);
        }

        template <typename T>
        int GEMM<T>::linearIndexF(int rows, int cols, int r, int c) {
            return (c * rows + r);
        }

        template <typename T>
        T* GEMM<T>::transpose(int orderSource, int orderTarget, int rows, int cols, T *source) {
            T *ret = new T[rows * cols];

            // handle transpose in parallel
#pragma omp parallel for proc_bind(close)
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int zIdx = orderTarget == CblasRowMajor ? linearIndexC(rows, cols, r, c) : linearIndexF(rows, cols, r, c);
                    int xIdx = orderSource == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

                    ret[zIdx] = source[xIdx];
                }
            }

            return ret;
        }

        template <typename T>
        void GEMM<T>::op(int Order, int TransA, int TransB,
                       int M, int N, int K,
                       T alpha,
                       T *A, int lda,
                       T *B, int ldb,
                       T beta,
                       T *C, int ldc) {

            // optionally handle transpose
            // we want C always here
            bool transAFlag = TransA != CblasTrans;
            bool transBFlag = TransB == CblasTrans;
//            T *aT = transAFlag ? transpose(CblasColMajor, CblasRowMajor, M, K, A) : A;

            // we want F always here
//            T *bT = transBFlag ? transpose(CblasRowMajor, CblasColMajor, K, N, B) : B;

            printf("Matrix multiply with: Order %c, TransA %c, TransB %c, Row count %i, Output columns %i, A/B column/rows %i (%s,%s)\n", Order, TransA, TransB, M, N, K, (transAFlag?"TRUE":"FALSE"), (transBFlag?"TRUE":"FALSE"));

            if (beta == (T) 0.0f) {
                int length = M*N;
                if (length <= 8192) {
#pragma omp simd
                    for (int r = 0; r < length; r++)
                        C[r] = (T) 0.0f;
                } else {
#pragma omp parallel for simd
                    for (int r = 0; r < length; r++)
                        C[r] = (T) 0.0f;
                }
            }


#pragma omp parallel for proc_bind(spread)
            for (int r = 0; r < M; r++) {
                if (!transAFlag && !transBFlag) { // matricies already in proper
                    int aIdx = linearIndexC(M, K, r, 0);
                    T *aX = A + aIdx;
    
                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);
    
                        T dot = (T) 0.0f;
    
                        if (alpha != (T) 0.0f) {
                            int bIdx = linearIndexF(K, N, 0, c);
    
                            T *bX = B + bIdx;
    
                            dot = nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                        }
    
                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }
                }
                else if (transAFlag && !transBFlag) { // matrix A need to be viewed as transposed              else {
                    int aIdx;// = linearIndexC(K, M, r, 0);
                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);
    
                        T dot = (T) 0.0f;
    
                        if (alpha != (T) 0.0f) {
                            int bIdx = linearIndexF(K, N, 0, c);
    
                            T *bX = B + bIdx;
                            //                    int aIdx = linearIndexC(K, M, r, 0);
                            for (int k = 0; k < K; k++) {
                                aIdx = linearIndexF(M, K, r, k);

                                dot += alpha * A[aIdx] * (bX[k]);//A[aIdx]nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                            }
                        }
    
                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }

                }
                else if (!transAFlag && transBFlag) { // matrix B need to be viewed as transposed
                    int aIdx = linearIndexC(M, K, r, 0);
                    T *aX = A + aIdx;

                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);
    
                        T dot = (T) 0.0f;
    
                        if (alpha != (T) 0.0f) {
                            int bIdx;// = linearIndexF(K, N, 0, c);
    
                            T *bX = B + bIdx;
                            //                    int aIdx = linearIndexC(K, M, r, 0);
                            for (int k = 0; k < K; k++) {
                                bIdx = linearIndexC(K, M, r, k);

                                dot += alpha * aX[k] * B[bIdx];//A[aIdx]nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                            }
                        }
    
                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }


                }
                else { // both matricies need to be viewed as transposed
                    int aIdx;
                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);
    
                        T dot = (T) 0.0f;
    
                        if (alpha != (T) 0.0f) {
                            int bIdx; // = linearIndexF(K, N, 0, c);
    
                            //T *bX = bT + bIdx;
                            //                    int aIdx = linearIndexC(K, M, r, 0);
                            for (int k = 0; k < K; k++) {
                                aIdx = linearIndexF(K, M, r, k);
                                bIdx = linearIndexC(K, N, k, c);
                                dot += alpha * A[aIdx] * B[k];//A[aIdx]nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                            }
                        }
    
                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }


                }

            }


            // if transpose was applied - dismiss transposed arrays
//            if (transAFlag)
//                delete[] aT;

//            if (transBFlag)
//                delete[] bT;
        }


        template<typename T>
        void GEMV<T>::op(int TRANS, int M, int N,
                       T alpha,
                       T* A,
                       int lda,
                       T* X,
                       int incx,
                       T beta,
                       T* Y,
                       int incy ) {

            T *aT = TRANS == CblasTrans ? GEMM<T>::transpose(CblasColMajor, CblasRowMajor, M, N, A) : A;

#pragma omp parallel for proc_bind(close)
            for (int r = 0; r < M; r++) {
                int aIdx = GEMM<T>::linearIndexC(M, N, r, 0);
                T *aX = aT + aIdx;


                T dot = nd4j::math::nd4j_dot<T>(aX, X, lda) * alpha;
                Y[r] =  beta == (T) 0.0f ? dot : dot + beta * Y[r];
            }

            if (TRANS == CblasTrans)
                delete[] aT;
        }


        template class ND4J_EXPORT GEMM<float>;
        template class ND4J_EXPORT GEMM<float16>;
        template class ND4J_EXPORT GEMM<double>;

        template class ND4J_EXPORT GEMV<float>;
        template class ND4J_EXPORT GEMV<float16>;
        template class ND4J_EXPORT GEMV<double>;
    }
}