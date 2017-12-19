//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/batched_gemm.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(batched_gemm, -1, -1, false, 0, 9) {
            int transA = INT_ARG(0);
            int transB = INT_ARG(1);
            int M = INT_ARG(2);
            int N = INT_ARG(3);
            int K = INT_ARG(4);
            int ldA = INT_ARG(5);
            int ldB = INT_ARG(6);
            int ldC = INT_ARG(7);
            int batchSize = INT_ARG(8);

            // basically A+B and 2 arrays of alpha and beta
            int expectedWidth = batchSize * 2 + 2;

            REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0, 0, "");
            REQUIRE_TRUE(block.width() == expectedWidth, 0, "BatchedGemm: expected number of input arrays is %i, but got %i instead", expectedWidth, block.width());

            auto alpha = INPUT_VARIABLE(0);
            auto beta = INPUT_VARIABLE(1);

            std::vector<NDArray<T>*> _A(batchSize);
            std::vector<NDArray<T>*> _B(batchSize);
            std::vector<NDArray<T>*> _C(batchSize);

            for(int e = 0; e < batchSize; e++) {
                _A[e] = INPUT_VARIABLE(e+2);
                _B[e] = INPUT_VARIABLE(e+2+batchSize);
                _C[e] = OUTPUT_VARIABLE(e);
            };

            REQUIRE_TRUE(_A.size() == _B.size() && _A.size() == _C.size() && _A.size() == batchSize, 0, "BatchedGemm: mismatched numbers of A, B, C for unknown reason");
            
            nd4j::ops::helpers::_bgemm<T>(_A, _B, _C, alpha, beta, M, N, K, ldA, ldB, ldC);
            
            return ND4J_STATUS_OK;
        };


        DECLARE_SHAPE_FN(batched_gemm) {
            auto shapeList = new ShapeList();
            int transA = INT_ARG(0);
            int transB = INT_ARG(1);
            int M = INT_ARG(2);
            int N = INT_ARG(3);
            int K = INT_ARG(4);
            int ldA = INT_ARG(5);
            int ldB = INT_ARG(6);
            int ldC = INT_ARG(7);
            int batchSize = INT_ARG(8);

            if (!(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0)) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;

                shapeList->push_back(newShape);
                return shapeList;
            }
            

            std::vector<int> shape({M, N});

            for (int e = 0; e < batchSize; e++) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

                shape::shapeBufferFortran(2, shape.data(), newShape);

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}