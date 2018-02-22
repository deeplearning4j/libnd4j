//
//  @author raver119@gmail.com
//

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(broadcast_dynamic_shape, 2, 1, false, 0, 0) {

            NDArray<T>* x_shape = INPUT_VARIABLE(0);
            NDArray<T>* y_shape = INPUT_VARIABLE(1);
            
            NDArray<T>* output = OUTPUT_VARIABLE(0);
    
            for (int e = 0; e < output->lengthOf(); e++) {
                (*output)(e) = nd4j::math::nd4j_max((*x_shape)(x++), (*y_shape)(y++));
            }
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(broadcast_dynamic_shape) {
            auto shapeList = new ShapeList(); 
            
            auto theFirst = inputShape->at(0);
            auto theSecond = inputShape->at(1);

            int* newshape;
            int shapeLength = nd4j::math::nd4j_max(theFirst->sizeAt(-1), theSecond->sizeAt(-1));
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(1), int);

            shape::shapeVector(shapeLength,  newshape);

            shapeList->push_back(newshape); 
            return shapeList;
        }

    }
}