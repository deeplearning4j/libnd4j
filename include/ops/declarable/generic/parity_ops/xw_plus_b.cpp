//
//  xw_plus_b op. Created by GS <george@skymind.io> 31.01.2018
//
//
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(xw_plus_b, 3, 1, false, 0, 0) {
            NDArray<T>* x = INPUT_VARIABLE(0);
            NDArray<T>* y = INPUT_VARIABLE(1);
            NDArray<T>* b = INPUT_VARIABLE(2);
            NDArray<T>* z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() <= 2 && y->rankOf() <= 2 && z->rankOf() <= 2, 0, "xw_plus_b: Input and Output NDArrays should have rank less or equal to 2");
            REQUIRE_TRUE(b->rankOf() == 1 && b->lengthOf() == x->rankOf(), 0, "xw_plus_b: Input vector should have proper dimension 1x%i. "
                "But %ix%i given.", x->rankOf(), b->rankOf(), b->lengthOf()) 

            // multiply x to y
            nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, T(1.0f), T(0.0f));

            // adding b vector
            z->addiRowVector(b);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(xw_plus_b) {
            int *inA = inputShape->at(0);
            int *inB = inputShape->at(1);
            int *shape;
            ALLOCATE(shape, block.getWorkspace(), 2, int);

            int *tmpA, *tmpB;
            COPY_SHAPE(inA, tmpA);
            COPY_SHAPE(inB, tmpB);


            int iSize = (int) block.getIArguments()->size();
            int transA = 0;
            int transB = 0;

            if (iSize > 0)
                transA = INT_ARG(0);

            if (iSize > 1)
                transB = INT_ARG(1);

            if (transA == 0)
                transA = 111;

            if (transB == 0)
                transB = 111;

            if (transA == 1)
                transA = 112;

            if (transB == 1)
                transB = 112;

            if (transA == 112)
                shape::transposeInplace(tmpA);

            if (transB == 112)
                shape::transposeInplace(tmpB);

            if (shape::rank(tmpA) == 1 && shape::isMatrix(tmpB)) {
                // special case here
                int *newShape;
                shape[0] = 1;
                shape[1] = tmpB[2];
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
                shape::shapeBufferFortran(2, shape, newShape);

                RELEASE(shape, block.getWorkspace());
                RELEASE(tmpA, block.getWorkspace());
                RELEASE(tmpB, block.getWorkspace());

                return new ShapeList(newShape);
            } else if (shape::isScalar(tmpA) && shape::isScalar(tmpB)) {
                // just scalar vs scalar
                shape[0] = 1;
                shape[1] = 1;
            }  else if (shape::isMatrix(tmpA) && shape::isVector(tmpB)) {
                // gemv case
                if (shape::rank(tmpB) == 2) {
                    shape[0] = tmpA[1];
                    shape[1] = tmpB[2];
                } else {
                    // we have new 1D shape here
                    int *newShape;
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
                    shape::shapeVector(tmpA[1], newShape);

                    RELEASE(shape, block.getWorkspace());
                    RELEASE(tmpA, block.getWorkspace());
                    RELEASE(tmpB, block.getWorkspace());

                    return new ShapeList(newShape);
                }
            } else if ((shape::isMatrix(tmpA) && shape::isMatrix(tmpB)) || (shape::isVector(tmpA) && shape::isMatrix(tmpB)) || (shape::isColumnVector(tmpA) && shape::isVector(tmpB))) {
                // gemm case
                shape[0] = tmpA[1];
                shape[1] = tmpB[2];
            } else if ((shape::isVector(tmpA) && shape::isScalar(tmpB)) || (shape::isScalar(tmpA) && shape::isVector(tmpB))) {
                // element-wise
                shape[0] = 1;
                shape[1] = (int) nd4j::math::nd4j_max<Nd4jIndex>(shape::length(tmpA), shape::length(tmpB));
            } else if (shape::isRowVector(tmpA) && shape::isRowVector(tmpB)) {
                // dot case
                shape[0] = 1;
                shape[1] = 1;
            }

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBufferFortran(2, shape, newShape);

            RELEASE(shape, block.getWorkspace());

            RELEASE(tmpA, block.getWorkspace());
            RELEASE(tmpB, block.getWorkspace());
            return new ShapeList(newShape);
        }
    }
}