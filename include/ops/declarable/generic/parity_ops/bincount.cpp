//
//  @author raver119@gmail.com
//

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(bincount, 1, 1, false, 0, 0)
            NDArray<T>* values = INPUT_VARIABLE(0);
            
            NDArray<T>* weights = nullptr; //INPUT_VARIABLE(1);
            if (block.width() > 1) {
                weights = INPUT_VARIABLE(1);
                REQUIRE_TRUE(input->isSameShape(weights), 0, "bincount: the input and weights shapes should be equals");
            }
            int maxLength = -1;
            int minLength = 0;
            int maxIndex = input->argMax();
            maxLength = int((*input)(maxIndex))  + 1;

            if (block.numI() > 0) {
                minLength = nd4j::math::max(INT_ARG(0), 0);
                if (block.numI() == 2) 
                    maxLength = nd4j::math::max(maxLenght, INT_ARG(1));
            }

            NDArray<T>* result = OUTPUT_VARIABLE(0);
            
            for (int e = 0; e < input->lengthOf(); e++) {
                int val = (*input)(e);
                if (val < maxLenght) {
                    (*result)(val)++;
                    if (weights != nullptr)
                        (*result)(val) += (*weights)(e);
                }
            }
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(bincount) {
            auto shapeList = new ShapeList(); 
            auto in = INPUT_VARIABLE(0);

            int maxIndex = input->argMax();
            int maxLength = int((*input)(maxIndex))  + 1;

            if (block.numI() > 1) 
                maxLength = nd4j::math::max(maxLenght, INT_ARG(1));

            int* newshape;
            
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(shapeRank), int);

            shape::shapeVector(maxLenght,  newshape);

            shapeList->push_back(newshape); 
            return shapeList;
        }

    }
}