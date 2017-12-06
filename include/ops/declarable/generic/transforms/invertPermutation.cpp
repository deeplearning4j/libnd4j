//
//  // created by Yurii Shyrma on 06.12.2017
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(invertPermutation, 1, 1, false, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* output = this->getZ(block);

    if(!input->isVector())
        throw "CONFIGURABLE_OP invertPermute: input array must be vector !";
    
    std::set<T> uniqueElems;
    const int lenght = input->lengthOf();
        
    for(int i = 0; i < lenght; ++i) {
        T elem  = (*input)(i);
        if(!uniqueElems.insert(elem).second)
            throw "CONFIGURABLE_OP invertPermute: input array contains duplicates !";
        if(elem < (T)0. || elem > lenght - (T)1.)
            throw "CONFIGURABLE_OP invertPermute: element of input array is out of range (0, lenght-1) !";
        (*output)((int)elem) = i;
    }
    
    return ND4J_STATUS_OK;
}
        



}
}
