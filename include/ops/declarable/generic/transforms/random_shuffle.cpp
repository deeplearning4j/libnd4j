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

    // generate randomly shuffle map of indexes to swap elements between    
    nd4j::random::RandomBuffer* rng = block.getRNG();    
    const int middle = firstDim / 2;
    std::unordered_set<int> shuffleSet;    
    while(shuffleSet.size() < middle)
        shuffleSet.insert(rng->nextInt(0, middle - 1));
    rng->rewind(middle);
    std::vector<int> shuffleVec(shuffleSet.begin(), shuffleSet.end());

    // evaluate sub-arrays of input array through all dimensions excluding first one
    std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {0});       
    ResultSet<T>* subArrsListIn = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);

    
    if(block.isInplace()) {

// #pragma omp parallel for schedule(guided)        
        for(int i = 0; i < middle; ++i)
            subArrsListIn->at(shuffleVec[i])->swapUnsafe(*(subArrsListIn->at(shuffleVec[i] + middle)));
        
        // if firstDim is odd then swap randomly middle sub-array
        if(firstDim % 2) {
            int randomForMiddle = rng->nextInt(0, firstDim - 1);
            if(randomForMiddle != middle)
                subArrsListIn->at(middle)->swapUnsafe(*(subArrsListIn->at(randomForMiddle))); 
            rng->rewind(1);
        }
    }
    else {
        
        ResultSet<T>* subArrsListOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);        

// #pragma omp parallel for schedule(guided)                
        for(int i = 0; i < middle; ++i) {
            subArrsListOut->at(shuffleVec[i])->assign(*(subArrsListIn->at(shuffleVec[i] + middle)));
            subArrsListOut->at(shuffleVec[i] + middle)->assign(*(subArrsListIn->at(shuffleVec[i])));
        }
     
        // if firstDim is odd then swap randomly middle sub-array
        if(firstDim % 2) {
            int randomForMiddle = rng->nextInt(0, firstDim - 1);
            if(randomForMiddle != middle)
                subArrsListOut->at(middle)->swapUnsafe(*(subArrsListOut->at(randomForMiddle))); 
            rng->rewind(1);
        }   
        
        delete subArrsListOut;
    }
            
    delete subArrsListIn;

    return Status::OK();
}


}
}