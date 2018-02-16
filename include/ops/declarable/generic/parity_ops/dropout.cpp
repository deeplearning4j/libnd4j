//
// Created by GS <sgazeos@gmail.com>
//


#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/dropout.h>

//#include <helpers/ShapeUtils.h>
//#include <vector>
//#include <numeric>


namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
//CONFIGURABLE_OP_IMPL(dropout, 3, 1, true, 0, 1) {
CONFIGURABLE_OP_IMPL(dropout, 1, 1, true, 1, 1) {
    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param
//    NDArray<T>* probability = INPUT_VARIABLE(1); // indeces, as is

    NDArray<T>* reduceShape = nullptr; //INPUT_VARIABLE(1);
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    
    int seed = INT_ARG(0);
    
    //REQUIRE_TRUE(probability->isScalar(), 0, "dropout: Need a scalar with range 0 to 1 as probability.");
    T probValue = T_ARG(0); //probability->getScalar(0);
    if (block.width() > 1)
        reduceShape = INPUT_VARIABLE(1);

    REQUIRE_TRUE(probValue > T(0.f) && probValue <= T(1.f), 0, "dropout: Probability should be with range 0 to 1.");

    if (probValue == T(1.0)) {
        *output = *input;
        return ND4J_STATUS_OK;
    }

    return helpers::dropOutFunctor(input, output, reduceShape, seed, probValue);
}

}
}