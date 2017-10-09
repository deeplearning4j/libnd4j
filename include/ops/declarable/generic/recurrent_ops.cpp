//
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
// @author iuriish@yahoo.com
//

#ifndef LIBND4J_RECURRENT_OPS_H
#define LIBND4J_RECURRENT_OPS_H

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
// feed forward
CONFIGURABLE_OP_IMPL(sru, 4, 2, true, 0, 0) {

    NDArray<T>* input   = INPUT_VARIABLE(0);                // input 3d tensor [KxbSxN]
    NDArray<T>* weights = INPUT_VARIABLE(1);                // 3d tensor of weights [KxbSx3N]
    NDArray<T>* biasF   = INPUT_VARIABLE(2);                // row of biases for forget gate [1×N]
    NDArray<T>* biasR   = INPUT_VARIABLE(3);                // row of biases for reset gate [1×N]
    
    NDArray<T> *activations = this->getZ(block);
    std::vector<int> argI = *(block.getIArguments());
    std::vector<T> argT = *(block.getTArguments());
    
    const int K  = input->shapeOf()[0];
    const int bS = input->shapeOf()[1];
    const int N  = input->shapeOf()[2];
    const int N3 = weights->shapeOf()[2];     // = 3*N
    const int ncols = bS*N;

    return ND4J_STATUS_OK;
}

 

}
}

#endif //LIBND4J_RECURRENT_OPS_H

