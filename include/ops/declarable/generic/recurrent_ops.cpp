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
CUSTOM_OP_IMPL(sru, 6, 2, false, 0, 0) {

    NDArray<T>* input   = INPUT_VARIABLE(0);                // input 3d tensor [K x bS x N]
    NDArray<T>* weights = INPUT_VARIABLE(1);                // 3d tensor of weights [K x bS x 3*N]
    NDArray<T>* biasF   = INPUT_VARIABLE(2);                // row of biases for forget gate [1 × N]
    NDArray<T>* biasR   = INPUT_VARIABLE(3);                // row of biases for reset gate [1 × N]
    NDArray<T>* prevState = INPUT_VARIABLE(4);              // 2d tensor of previous state ????? [1, bS*N] or [K x bS x N] ??????
    NDArray<T>* mask    = INPUT_VARIABLE(5);                // dropout mask

    NDArray<T>* curState = OUTPUT_VARIABLE(0);             // c_t, [K x bS x N]
    NDArray<T>* output   = OUTPUT_VARIABLE(1);             // h_t,   [K x bS x N]
    
    const int K       = input->shapeOf()[0];
    const int bS      = input->shapeOf()[1];
    const int N       = input->shapeOf()[2];
    const int ncols_i = bS*N;
    const int ncols_w = ncols_i*3;
  
    T* const pInput   = input->getBuffer();
    T* const pWeights = weights->getBuffer();
    T* const pBiasF   = biasF->getBuffer();
    T* const pBiasR   = biasR->getBuffer();
    T* const pPrev    = prevState->getBuffer();
    T* const pMask    = mask->getBuffer();
    T* const pCur     = curState->getBuffer();
    T* const pOut     = output->getBuffer();
    
    T bF, bR, msk, cur, val, g1, g2, *pW(nullptr), *pIn(nullptr), *pC(nullptr), *pO(nullptr);
    for (int col = 0; col < ncols_i; ++col) {    
        bF  = *(pBiasF + col%N);
        bR  = *(pBiasR + col%N);
        msk = *(pMask + col);
        cur = *(pPrev + col);
        pW  = pWeights + 3*col;
        pIn = pInput + col;
        pC  = pCur + col;
        pO  = pOut + col;

        for (int row = 0; row < K; ++row) {
            g1 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+1) + bF)));
            g2 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+2) + bR)));
            cur = g1*cur + ((T)1. - g1)*(*pW);
            *pC = cur;
            // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
            val = nd4j::math::nd4j_tanh<T>(cur);
            *pO = val*msk*g2 - ((T)1. - g2)*(*pIn);
            pW  += ncols_w;
            pIn += ncols_i;
            pC  += ncols_i;
            pO  += ncols_i;
        }
    }
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru) {

    int* inShape = inputShape->at(0);
    int rank = inShape[0];          // = 3
    int size = rank*2 + 4;
    int K  = inShape[1];
    int bS = inShape[2];
    int N  = inShape[3];
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = K;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   

//////////////////////////////////////////////////////////////////////////
// feed forward bidirectional 
CUSTOM_OP_IMPL(sru_bi, 5, 2, false, 0, 0) {

    NDArray<T>* weights   = INPUT_VARIABLE(0);                // 3d tensor of weights [K x bS x 3*N]
    NDArray<T>* biasF     = INPUT_VARIABLE(1);                // row of biases for forget gate [1 × N]
    NDArray<T>* biasR     = INPUT_VARIABLE(2);                // row of biases for reset gate [1 × N]
    NDArray<T>* prevState = INPUT_VARIABLE(3);                // 2d tensor of previous state ????? [1, bS*N] or [K x bS x N] ??????
    NDArray<T>* mask      = INPUT_VARIABLE(4);                // dropout mask

    NDArray<T>* curState = OUTPUT_VARIABLE(0);             // c_t, [K x bS x 2N]
    NDArray<T>* output   = OUTPUT_VARIABLE(1);             // h_t,   [K x bS x 2N]

    const int K       = weights->shapeOf()[0];
    const int bS      = weights->shapeOf()[1];
    const int N       = biasF->shapeOf()[1];
    const int N2      = N*2;
    const int ncols_i = bS*N2;
    const int ncols_w = ncols_i;
    const int len     = K;
  
    T* const pWeights = weights->getBuffer();
    T* const pBiasF   = biasF->getBuffer();
    T* const pBiasR   = biasR->getBuffer();
    T* const pPrev    = prevState->getBuffer();
    T* const pMask    = mask->getBuffer();
    T* const pCur     = curState->getBuffer();
    T* const pOut     = output->getBuffer();
    
    T bF, bR, msk, cur, val, g1, g2, *pW(nullptr), *pIn(nullptr), *pC(nullptr), *pO(nullptr);
    bool flip;
    for (int col = 0; col < ncols_i; ++col) {    
        
        msk = *(pMask + col);
        cur = *(pPrev + col);
        bF  = *(pBiasF + col%N2);
        bR  = *(pBiasR + col%N2);
        pW  = pWeights + col;
        pIn = pW + 3;        
        pC  = pCur + col;
        pO  = pOut + col;
        
        flip = (col%N2) >= N;        
        if (flip) {
            pW  += (len-1)*ncols_w;
            pIn += (len-1)*ncols_i;
            pC  += (len-1)*ncols_i;
            pO  += (len-1)*ncols_i;
        }

        for (int row = 0; row < K; ++row) {
            g1 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+1) + bF)));
            g2 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+2) + bR)));
            cur = g1*cur + ((T)1. - g1)*(*pW);
            *pC = cur;
            // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
            val = nd4j::math::nd4j_tanh<T>(cur);
            *pO = val*msk*g2 - ((T)1. - g2)*(*pIn);
            pW  += ncols_w;
            pIn += ncols_i;
            pC  += ncols_i;
            pO  += ncols_i;
        }
    }
        return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bi) {

    int* inShape = inputShape->at(3);
    int rank = inShape[0];          // = 3
    int size = rank*2 + 4;
    int K  = inShape[1];
    int bS = inShape[2];
    int N  = inShape[3];
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = K;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = 2*N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   


 

}
}

#endif //LIBND4J_RECURRENT_OPS_H

