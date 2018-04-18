//
//  @author raver119@gmail.com
//

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/top_k.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(in_top_k, 2, 1, true, 0, 1) {
            NDArray<T>* predictions = INPUT_VARIABLE(0);
            NDArray<T>* target = INPUT_VARIABLE(1);

            NDArray<T>* result = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(block.numI() > 0, 0, "in_top_k: Parameter k is needed to be set");
            REQUIRE_TRUE(predictions->sizeAt(0) == target->sizeAt(0), 0, "in_top_k: The predictions and target should have equal number of columns");
            REQUIRE_TRUE(predictions->rankOf() == 2, 0, "in_top_k: The predictions array shoud have rank 2, but %i given", predictions->rankOf());
            REQUIRE_TRUE(target->rankOf() == 1, 0, "in_top_k: The target should be a vector");

            int k = INT_ARG(0);
            return helpers::inTopKFunctor(predictions, target, result, k);
        }

        DECLARE_SHAPE_FN(in_top_k) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(1);
            int shapeRank = shape::rank(in);

            int* newshape;

            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(shapeRank), int);

//            if (shape::order(in) == 'c')
//                shape::shapeVector(shape::sizeAt(in, 0),  newshape);
//            else
//                shape::shapeVectorFortran(shape::sizeAt(in, 0),  newshape);
            if (shape::order(in) == 'c')
                shape::shapeBuffer(shape::rank(in), shape::shapeOf(in), newshape);
            else 
                shape::shapeBufferFortran(shape::rank(in), shape::shapeOf(in), newshape);

            shapeList->push_back(newshape); 
            return shapeList;
        }

    }
}