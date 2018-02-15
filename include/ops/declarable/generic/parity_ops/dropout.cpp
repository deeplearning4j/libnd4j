//
// Created by GS <sgazeos@gmail.com>
//

#include <NativeOps.h>
#include <ops/declarable/headers/parity_ops.h>

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


    std::vector<Nd4jIndex> buffer(100000);
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, seed, buffer.size(), (Nd4jPointer) &buffer[0]);

    if (rng == nullptr)
        return ND4J_STATUS_BAD_RNG;



    if (reduceShape == nullptr)
        input->template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, output, &probValue);
    else {
        REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");
    
        std::vector<int> dims(reduceShape->lengthOf());
    
        bool fit = true;
    
        for( int i = 0; i < dims.size(); i++ ) {
            dims[i] = (*reduceShape)(i);
            for (int e = 0; e < input->rankOf(); ++e)
                if (input->sizeAt(e) % dims[i]) {
                    fit = false;
                    break;
                }
    
            if(!fit) break;
        }
    
        // check dims to fit input
        REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
        NDArray<T> chunk('c', dims);
        chunk.assign(T(1.0));
        chunk.template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, &chunk, &probValue);
    
        // broadcast chunk to full matrix
        std::unique_ptr<NDArray<T>> dropOutMultiplier(new NDArray<T>(*input));
        dropOutMultiplier->assign(T(0.0));
    
        *dropOutMultiplier += chunk;
    
        input->template applyPairwiseTransform<simdOps::Multiply<T>>(dropOutMultiplier.get(), output, nullptr);
    }
    nativeOps.destroyRandom(rng);

    return ND4J_STATUS_OK;
}

}
}