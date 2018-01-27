//
// Created by Yurii Syrma on 26.01.2018
//

#include <ops/declarable/CustomOperations.h>
#include <unordered_set>

namespace nd4j {
namespace ops {

OP_IMPL(random_shuffle, 1, 1, true) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = nullptr;
    if(!block.isInplace())
       output = OUTPUT_VARIABLE(0);
    
    REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RANDOM_SHUFFLE op: RNG should be defined in Graph !");

    // check edge cases first
    const int firstDim = input->sizeAt(0);    
    if(input->lengthOf() == 1 || firstDim == 1) {
        
        if(!block.isInplace())
            output->assign(input);
        
        return Status::OK();
    }
    
    // get instance of random generator
    nd4j::random::RandomBuffer* rng = block.getRNG();    
    
    // evaluate sub-arrays list of input array through all dimensions excluding first one
    std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {0});       
    ResultSet<T>* subArrsListIn = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);

    // apply Fisher-Yates shuffle
    if(block.isInplace()) {
// #pragma omp parallel for schedule(guided)        
        for(int i = firstDim-1; i > 0; --i) {
            int r = rng->nextInt(0, i);
            if(i == r)
                continue;
            subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
        }        
    }
    else {
        // evaluate sub-arrays list of output array through all dimensions excluding first one        
        ResultSet<T>* subArrsListOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);        
        // in not-in-place case we have to check whether zero element was shuffled
        bool isZeroElemShuffled = false;
// #pragma omp parallel for schedule(guided)        
        for(int i = firstDim-1; i > 0; --i) {
            int r = rng->nextInt(0, i);            
            if(r == 0)
                isZeroElemShuffled = true;
            subArrsListOut->at(i)->assign(subArrsListIn->at(r));
            if(i == r)
                continue;
            subArrsListOut->at(r)->assign(subArrsListIn->at(i));
        }           

        if(!isZeroElemShuffled)
            subArrsListOut->at(0)->assign(subArrsListIn->at(0));

        delete subArrsListOut;
    }
    
    rng->rewindH(firstDim-1);

    delete subArrsListIn;

    return Status::OK();
}


}
}