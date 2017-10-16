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
CUSTOM_OP_IMPL(sru1, 5, 2, false, 0, 0) {

    NDArray<T>* input     = INPUT_VARIABLE(0);                // input 3d tensor [K x bS x N]
    NDArray<T>* weights   = INPUT_VARIABLE(1);                // 3d tensor of weights [K x bS x 3*N]
    NDArray<T>* bias      = INPUT_VARIABLE(2);                // row of biases with twice length [1 × 2*N]
    NDArray<T>* prevState = INPUT_VARIABLE(3);                // 2d tensor of previous state ????? [1, bS*N] or [K x bS x N] ??????
    NDArray<T>* mask      = INPUT_VARIABLE(4);                // dropout mask

    NDArray<T>* output   = OUTPUT_VARIABLE(0);                // h_t,   [K x bS x N]
    NDArray<T>* curState = OUTPUT_VARIABLE(1);                // c_t, [K x bS x N]
    
    const int K       = input->shapeOf()[0];
    const int bS      = input->shapeOf()[1];
    const int N       = input->shapeOf()[2];
    const int ncols_i = bS*N;
    const int ncols_w = ncols_i*3;
  
    T* const pInput   = input->getBuffer();
    T* const pWeights = weights->getBuffer();
    T* const pBias    = bias->getBuffer();
    T* const pPrev    = prevState->getBuffer();
    T* const pMask    = mask->getBuffer();
    T* const pCur     = curState->getBuffer();
    T* const pOut     = output->getBuffer();
    
    T bF, bR, msk, cur, val, g1, g2, *pW(nullptr), *pIn(nullptr), *pC(nullptr), *pO(nullptr);       
    for (int col = 0; col < ncols_i; ++col) {           
        bF  = *(pBias + col%N);        // forget gate
        bR  = *(pBias + col%N + N);    // reset gate
        msk = *(pMask + col);        
        cur = *(pPrev + col);
        pW  = pWeights + 3*col;
        pIn = pInput + col;
        pC  = pCur + col;
        pO  = pOut + col;

        for (int row = 0; row < K; ++row) {
            g1 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+1) + bF)));
            g2 = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pW+2) + bR)));
            cur = (cur - *pW)*g1 + *pW;
            *pC = cur;
            // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
            val = nd4j::math::nd4j_tanh<T>(cur);
            *pO = (val*msk - *pIn)*g2 + *pIn;
            pW  += ncols_w;
            pIn += ncols_i;
            pC  += ncols_i;
            pO  += ncols_i;            
        }
    }
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru1) {

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
// feed forward, https://github.com/musyoku/chainer-sru/blob/master/sru/sru.py
CUSTOM_OP_IMPL(sru2, 5, 2, false, 0, 0) {

    NDArray<T>* input   = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
    NDArray<T>* bias    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
    NDArray<T>* init    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0
    NDArray<T>* mask    = INPUT_VARIABLE(4);                // 2d tensor of dropout mask [bS x K]

    NDArray<T>* output = OUTPUT_VARIABLE(0);                // h_t, [bS x K x N]
    NDArray<T>* state  = OUTPUT_VARIABLE(1);                // c_t, [bS x K x N]
    
    const int bS     = input->shapeOf()[0];                     // bS - batch size
    const int K      = input->shapeOf()[1];                     // K - number of features
    const int N      = input->shapeOf()[2];                     // N - number of time steps
    
    // multiplication matrix = matmul(weights,input)
    NDArray<T>* wi = NDArrayFactory<T>::mmulHelper(weights, input, nullptr, (T)1., (T)0.);      //        [bS x 3K x N]
    // wi.printShapeInfo();
    NDArray<T>* wiZ = wi->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } );       // [bS x K x N]
    NDArray<T>* wiF = wi->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } );       // forget gate [bS x K x N]
    NDArray<T>* wiR = wi->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } );       // reset gate [bS x K x N]
    NDArray<T>* bF  = bias->subarray( { NDIndex::all(), NDIndex::interval(0,K)  } );                        // biases for forget gate [1 x K]
    NDArray<T>* bR  = bias->subarray( { NDIndex::all(), NDIndex::interval(K,2*K)} );                        // biases for reset gate [1 x K]

    NDArray<T>* xt   = nullptr;
    NDArray<T>* zt   = nullptr;
    NDArray<T>* ft   = nullptr;
    NDArray<T>* rt   = nullptr;
    NDArray<T>* ct   = nullptr;
    NDArray<T>* ht   = nullptr;
    NDArray<T>* xmt  = new NDArray<T>(input->ordering(), {bS, K});
    NDArray<T>* gct  = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* ct_1 = init->dup(init->ordering());

    for (int t = 0; t < N; ++t) {           

        xt = input->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );   // [bS x K x 1]
        xt->reshapei(xt->ordering(), {bS, K});                                                  // [bS x K]
        zt = wiZ->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );     // [bS x K x 1]
        zt->reshapei(zt->ordering(), {bS, K});                                                  // [bS x K]
        ft = wiF->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );     // forget gate [bS x K x 1]
        ft->reshapei(ft->ordering(), {bS, K});                                                  // [bS x K]
        rt = wiR->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );     // reset gate [bS x K x 1]
        rt->reshapei(rt->ordering(), {bS, K});                                                  // [bS x K]
        ct = state->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );   // [bS x K x 1]
        ct->reshapei(ct->ordering(), {bS, K});                                                  // [bS x K]
        ht = output->subarray( { NDIndex::all(), NDIndex::all(), NDIndex::interval(t,t+1) } );  // [bS x K x 1]
        ht->reshapei(ht->ordering(), {bS, K});                                                  // [bS x K]

        //  xmt = xt * mask
        xt->template applyPairwiseTransform<simdOps::Multiply<T>>(mask, xmt, nullptr);
        // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
        ft->addRowVector(bF, ft);
        rt->addRowVector(bR, rt);
        ft->template applyTransform<simdOps::Sigmoid<T>>();
        rt->template applyTransform<simdOps::Sigmoid<T>>();
        // ct = ft * c_t-1 + (1 - ft) * zt,  
        ft->template applyPairwiseTransform<simdOps::Multiply<T>>(ct_1, ct, nullptr);
        ft->template applyTransform<simdOps::Neg<T>>();
        ft->template applyScalar<simdOps::Add<T>>((T)1., nullptr);
        ft->template applyPairwiseTransform<simdOps::Multiply<T>>(zt, nullptr);
        ct->template applyPairwiseTransform<simdOps::Add<T>>(ft, nullptr);

        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct->template applyTransform<simdOps::Tanh<T>>(gct);
        
        // ht = rt * gct + (1 - rt) * xt
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(gct, ht, nullptr);
        rt->template applyTransform<simdOps::Neg<T>>();
        rt->template applyScalar<simdOps::Add<T>>((T)1., nullptr);
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(xt, nullptr);
        ht->template applyPairwiseTransform<simdOps::Add<T>>(rt, nullptr);

        delete xt; delete zt; delete ft; delete rt; delete ht; delete ct_1;
        ct_1 = ct;
    }
            
    delete wiZ;
    delete wiF;
    delete wiR;
    delete wi;
    delete bF;
    delete bR;    
    delete xmt;
    delete ct_1;
    delete gct;
    
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru2) {

    int* inShape = inputShape->at(0);   // [bS x K x N]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   


//////////////////////////////////////////////////////////////////////////
// feed forward bidirectional 
CUSTOM_OP_IMPL(sru_bi, 4, 2, false, 0, 0) {

    NDArray<T>* weights   = INPUT_VARIABLE(0);                // 3d tensor of weights [K x bS x 3*N]
    NDArray<T>* bias      = INPUT_VARIABLE(1);                // row of biases with twice length [1 × 2*N]
    NDArray<T>* prevState = INPUT_VARIABLE(2);                // 2d tensor of previous state ????? [1, bS*N] or [K x bS x N] ??????
    NDArray<T>* mask      = INPUT_VARIABLE(3);                // dropout mask

    NDArray<T>* output   = OUTPUT_VARIABLE(0);             // h_t,  [K x bS x 2N]
    NDArray<T>* curState = OUTPUT_VARIABLE(1);             // c_t,  [K x bS x 2N]    

    const int K       = prevState->shapeOf()[0];
    const int bS      = prevState->shapeOf()[1];
    const int N       = prevState->shapeOf()[2];
    const int N2      = N*2;
    const int ncols_i = bS*N2;
    const int ncols_w = ncols_i;
    const int len     = K;
  
    T* const pWeights = weights->getBuffer();
    T* const pBiasF   = bias->getBuffer();
    T* const pBiasR   = bias->getBuffer();
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
        bR  = *(pBiasR + col%N2 + N2);
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
            *pO = (val*msk - *pIn)*g2 + *pIn;
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

