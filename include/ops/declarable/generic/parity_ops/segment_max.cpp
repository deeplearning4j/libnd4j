//
// Created by george@skymind.io on 2/21/2018.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(segment_max, 2, 1, false, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* idxSegments = INPUT_VARIABLE(1);
            NDArray<T>* segmentedOutput = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(idxSegments->isVector(), 0, "segment_max: segment indexes array should be a vector, but it rank is %i.", idxSegments->rankOf());
            REQUIRE_TRUE(idxSegments->lengthOf() == input->sizeAt(0), 0, "segment_max: segment indexes array length should be equal to the input first dimension, but %i != %i.", idxSegments->lengthOf(), input->sizeAt(0));
            int numClasses = segmentedOutput->sizeAt(0);
            int pos = 0;
            // if input is a vector: (as if in doc sample)
            if (input->isVector()) {
                T val = (*input)(0);
                T idx = (*idxSegments)(0);
                for (int e = 1; e < idxSegments->lengthOf(); e++) {
                    if (idx == (*idxSegments)(e)) {
                       // max 
                       val = nd4j::math::nd4j_max(val, (*input)(e));
                    }
                    else {
                        pos++;
                        idx = (*idxSegments)(e);
                        val = (*input)(e);
                    }
                    (*segmentedOutput)(pos) = val;
                }
            }
            else if (input->isMatrix()) {
                T idx = (*idxSegments)(0);
                int stalled = 1;
                int firstRow = 0;
                T val = (*input)(firstRow, 0);

                for (int e = 1; e < idxSegments->lengthOf(); e++) {
                    if (idx == (*idxSegments)(e)) {
                       // max 
                        stalled++;
                        for (int k = 0; k < stalled; k++)
                            val = nd4j::math::nd4j_max(val, (*input)(firstRow + k, e));
                    }
                    else {
                        idx = (*idxSegments)(e);
                        val = (*input)(firstRow, e);
                        firstRow += stalled;
                        stalled = 1;
                        pos++;
                    }
                    (*segmentedOutput)(pos) = val;

                }
            }
            else {
                
            }
            //
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(segment_max) {

            NDArray<T>* idxVector = INPUT_VARIABLE(1);
//            std::vector<int> outputShape;
            int* in = inputShape->at(0);
            int numOfClasses = 1; //shape::sizeAt(in, 0);
            int outRank = shape::rank(in);
            int* outputShape = nullptr;
            T val = (*idxVector)(0);
            for (int e = 1; e < idxVector->lengthOf(); e++) {
                if (val != (*idxVector)(e)) {
                    numOfClasses++;
                    val = (*idxVector)(e);
                }
            }
            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), int);

            outputShape[0] = outRank;
            outputShape[1] = numOfClasses;
            for(int i = 1; i < outRank; ++i)
                outputShape[i + 1] = shape::sizeAt(in, i);

            shape::updateStrides(outputShape, shape::order(in));

            return SHAPELIST(outputShape);
        }
    }

}
