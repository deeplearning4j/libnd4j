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

            // M - row count for A
            // N - column count for B
            // K - column count for A
            // C matrix has MxN dim
            // optionally handle transpose
            // we want C always here
            bool transposeAFlag = TransA != 'N' && TransA != 'n';
            bool transposeBFlag = TransB != 'N' && TransB != 'n';
            //T *aT = transposeAFlag ? transpose(CblasColMajor, CblasRowMajor, M, K, A) : A;

            // we want F always here
            //T *bT = transposeBFlag ? transpose(CblasRowMajor, CblasColMajor, K, N, B) : B;


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

            int rowCount = (transposeAFlag?K:M);
            int rowCountB = (transposeBFlag?N:K);
            int columnCountA = (transposeAFlag?M:K);
            int columnCount = (transposeBFlag?K:N);
            int strideA = (transposeAFlag?columnCountA:1);
            int strideB = (transposeBFlag?1:columnCount);

            printf("Row count %i, Column count %i, B rows %i, A columns %i, lda %i, ldb %i\n", rowCount, columnCount, rowCountB, columnCountA, lda, ldb);
            printf("strideA %i, strideB %i\n", strideA, strideB);
            for (int i = 0; i < rowCount; i++) {
                for (int j = 0; j < columnCountA; j++)
                    printf("%010.2lf  ", (double)A[i * K + j]);
                printf("\n");
            }
            if (rowCountB != columnCountA) {
                printf("Something wrong with input matricies: %i != %i\n", columnCountA, rowCountB);
            }
            printf("\n==========================================\n");
            for (int i = 0; i < rowCountB; i++) {
                for (int j = 0; j < columnCount; j++)
                    printf("%010.2lf  ", (double)B[i * N + j]);
                printf("\n");
            }

#pragma omp parallel for proc_bind(spread)
            for (int r = 0; r < rowCount; r++) {

                int aIdx = 0; //linearIndexC(rowCount, columnCount, r, 0);
                //T *aX = aT + aIdx;
                 
                printf("r = %i, aIdx = %i from %i\n", r, aIdx, rowCount * columnCountA);
                
                for (int c = 0; c < columnCount; c++) {
                    int zIdx = linearIndexF(rowCount, columnCount, r, c);

                    T dot = (T) 0.0f;

                    if (alpha != (T) 0.0f) {
                        int bIdx = 0; //-strideB; //linearIndexF(columnCount, columnCount, 0, c);
                        printf("c = %i, bIdx = %i from %i\n", c, bIdx, rowCountB * columnCount);

                        //T *bX = bT + bIdx;
                        for (int e = 0; e < rowCountB; e++) {
                            aIdx = e * strideA + r;
                            bIdx = e * strideB + c;// * e;
                            printf("%i: aIdx = %i, bIdx = %i\n", e, aIdx, bIdx);

                            dot += A[aIdx] * B[bIdx];
//                            dot = nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                        }
                    }
                    if (beta != (T) 0.0f) {
                        C[zIdx] = dot + beta * C[zIdx];
                    } else {
                        C[zIdx] = dot;
                    }
                }
            }


            // if transpose was applied - dismiss transposed arrays
//            if (transposeAFlag)
//                delete[] aT;

//            if (transposeBFlag == CblasTrans)
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
