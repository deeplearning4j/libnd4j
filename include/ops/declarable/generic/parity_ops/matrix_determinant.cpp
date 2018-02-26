//
// Created by GS <sgazeos@gmail.com> at 2/26/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lup.h>
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matrix_determinant, 1, 1, false, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(1);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "matrix_determinant: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "matrix_determinant: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));
            int n = input->sizeAt(-1);

            for (int e = 0; e < output->lengthOf(); e++) {
                IndicesList indx;
                indx.push_back(NDIndex::interval(e * n, e * n +  n * n));
                NDArray<T>* matrix = input->subarray(indx);

                (*output) = helpers::lup(matrix, (NDArray<T>*)nullptr, (NDArray<T>*)nullptr);

                delete matrix;
            }
            
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(tear) {
            auto inShape = inputShape->at(0);

            int* determinantShape;
            int targetRank = shape::rank(inShape) - 2; // last two dimensions will be reduced to scalar

            if (targetRank == 0) { // scalar only
                determinantShape = shape::createScalarShapeInfo();
            }
            else if (targetRank == 1) { // vector 
                ALLOCATE(determinantShape, block.getWorkspace(), shape::shapeInfoLength(targetRank), int);
                shape::shapeVector(shape::sizeAt(inShape, 0), determinantShape);
            }
            else { // only two last dimensions are excluded
                ALLOCATE(determinantShape, block.getWorkspace(), shape::shapeInfoLength(targetRank), int);

                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(targetRank, shape::shapeOf(inShape), determinantShape);
                else
                    shape::shapeBufferFortran(targetRank, shape::shapeOf(inShape), determinantShape);
            }
            return new ShapeList(determinantShape);
        }
    }
}