//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <types/float16.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <helpers/BlasHelper.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
        


            template <typename T>
            void _bgemm(std::vector<NDArray<T>*>& vA, std::vector<NDArray<T>*>& vB, std::vector<NDArray<T>*>& vC, NDArray<T>* alphas, NDArray<T>* betas, int M, int N, int K, int ldA, int ldB, int ldC) {
                if (BlasHelper::getInstance()->hasBatchedGEMM<T>()) {
                    if (sizeof(T) == 8) {
                        
                    } else if (sizeof(T) == 4) {

                    }
                } else {

#pragma omp parallel for                   
                    for (int p = 0; p < vA.size(); ++p) {
                        auto A = vA.at(p)->buffer();
                        auto B = vB.at(p)->buffer();
                        auto C = vC.at(p)->buffer();
                        auto alpha = alphas->getScalar(p);
                        auto beta = betas->getScalar(p);
                        for (int m = 0; m < M; ++m) {
                            for (int n = 0; n < N; ++n) {
                                T c_mnp = 0;

                                #pragma omp simd
                                for (int k = 0; k < K; ++k)
                                    c_mnp += A[m + k * ldA] * B[k + n * ldB];

                                C[m + n * ldC] = alpha * c_mnp + beta * C[m + n * ldC];
                            } 
                        } 
                    }
                }
            };

            template void _bgemm<float>(std::vector<NDArray<float>*>& vA, std::vector<NDArray<float>*>& vB, std::vector<NDArray<float>*>& vC, NDArray<float>* alphas, NDArray<float>* betas, int M, int N, int K, int ldA, int ldB, int ldC);
            template void _bgemm<double>(std::vector<NDArray<double>*>& vA, std::vector<NDArray<double>*>& vB, std::vector<NDArray<double>*>& vC, NDArray<double>* alphas, NDArray<double>* betas, int M, int N, int K, int ldA, int ldB, int ldC);
            template void _bgemm<float16>(std::vector<NDArray<float16>*>& vA, std::vector<NDArray<float16>*>& vB, std::vector<NDArray<float16>*>& vC, NDArray<float16>* alphas, NDArray<float16>* betas, int M, int N, int K, int ldA, int ldB, int ldC);
        }
    }
}