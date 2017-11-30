//
// implementation of operation for conventional LSTM cell: S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
//
// created by Yurii Shyrma on 30.11.2017
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> actvation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void clipping(NDArray<T>& arr, T limit) {    
    
    if(limit < (T)0.)
    	limit *= (T)(-1.);

    auto clip = LAMBDA_T(value, limit) {
    	if(value < -limit || value > limit)
    		value = limit;
    	return value; 
 	};

    arr.applyLambda(clip);    
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmCell, 6, 2, false, 1, 0) {

	NDArray<T> xt   = *INPUT_VARIABLE(0);					// input [batchSize x inSize]
	NDArray<T> ht_1 = *INPUT_VARIABLE(1);					// previous cell output [batchSize x numUnits], that is at previous time step t-1
	NDArray<T> ct_1 = *INPUT_VARIABLE(2);					// previous cell state  [batchSize x numUnits], that is at previous time step t-1	

	NDArray<T> Wx   = *INPUT_VARIABLE(3);					// input-to-hidden  weights,  [inSize   x 4*numUnits]	
	NDArray<T> Wh   = *INPUT_VARIABLE(4);					// hidden-to-hidden weights,  [numUnits x 4*numUnits]	
	NDArray<T> b    = *INPUT_VARIABLE(5);					// biases, [4 x numUnits] 
	
	NDArray<T> ht   = *OUTPUT_VARIABLE(0);      			// current cell output [batchSize x numUnits], that is at current time step t
    NDArray<T> ct   = *OUTPUT_VARIABLE(1);      			// current cell state  [batchSize x numUnits], that is at current time step t

    const int numUnits  = ht_1.sizeAt(1);

    // input gate = sigmoid(mmul(Wxi,xt) + mmul(Whi,ht_1) + bi)
    NDArray<T> Wxi = Wx({{},{0, numUnits}});
    NDArray<T> Whi = Wh({{},{0, numUnits}});
    NDArray<T> bi  = b({{0,1},{}});
    NDArray<T> it  = sigmoid<T>(mmul(xt, Wxi) + mmul(ht_1, Whi) + bi);		
    
    // forget gate = sigmoid(mmul(Wxf,xt) + mmul(Whf,ht_1) + bf)
    NDArray<T> Wxf = Wx({{},{numUnits, 2*numUnits}});
	NDArray<T> Whf = Wh({{},{numUnits, 2*numUnits}});
	NDArray<T> bf  = b({{1,2},{}});
    NDArray<T> ft = sigmoid<T>(mmul(xt, Wxf) + mmul(ht_1, Whf) + bf);		

    // output gate = sigmoid(mmul(Wxo,xt) + mmul(Who,ht_1) + bo)
    NDArray<T> Wxo = Wx({{},{2*numUnits, 3*numUnits}});
	NDArray<T> Who = Wh({{},{2*numUnits, 3*numUnits}});
	NDArray<T> bo  = b({{2,3},{}});
    NDArray<T> ot = sigmoid<T>(mmul(xt, Wxo) + mmul(ht_1, Who) + bo);

    // current sell state = ft*ct_1 + it*actvation(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc)
    NDArray<T> Wxc = Wx({{},{3*numUnits, 4*numUnits}});
	NDArray<T> Whc = Wh({{},{3*numUnits, 4*numUnits}});
	NDArray<T> bc  = b({{3,4},{}});
    ct = ft * ct_1 + it * actvation<T>(mmul(xt, Wxc) + mmul(ht_1, Whc) + bc);

    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(!block.getTArguments()->empty())
    	clipping(ct, T_ARG(0));

    // current cell output = ot*actvation(ct)
    ht = ot * actvation<T>(ct);

    STORE_2_RESULTS(ht, ct);

}



DECLARE_SHAPE_FN(lstmCell) {

    const int batchSize = (INPUT_VARIABLE(0))->sizeAt(0);
    const int inSize    = (INPUT_VARIABLE(0))->sizeAt(1);
    const int numUnits  = (INPUT_VARIABLE(1))->sizeAt(1);

    // check shapes of previous cell output and previous cell state
    for(int i = 1; i <=2; ++i)
    	if(!INPUT_VARIABLE(i)->isSameShape({batchSize, numUnits}));
    		throw "CUSTOM_OP lstmCell: the shape of previous cell output or previous cell state is wrong !";
    
    // check shape of input-to-hidden  weights
    if(!INPUT_VARIABLE(3)->isSameShape({inSize, 4*numUnits}));
    	throw "CUSTOM_OP lstmCell: the shape of input-to-hidden weights is wrong !";

    // check shape of hidden-to-hidden  weights
    if(!INPUT_VARIABLE(4)->isSameShape({numUnits, 4*numUnits}));
    	throw "CUSTOM_OP lstmCell: the shape of hidden-to-hidden weights is wrong !";

    // check shape of biases
    if(!INPUT_VARIABLE(5)->isSameShape({4, numUnits}));
    	throw "CUSTOM_OP lstmCell: the shape of biases is wrong !";

    // evaluate output shapeInfos
   	int *outShapeInfo1(nullptr), *outShapeInfo2(nullptr);
    ALLOCATE(outShapeInfo1, block.getWorkspace(), 8, int);
    ALLOCATE(outShapeInfo2, block.getWorkspace(), 8, int);
            
    outShapeInfo1[0] = outShapeInfo2[0] = 2;
    outShapeInfo1[1] = outShapeInfo2[1] = batchSize;
    outShapeInfo1[2] = outShapeInfo2[2] = numUnits;
    
    shape::updateStrides(outShapeInfo1, (INPUT_VARIABLE(1))->ordering());
    shape::updateStrides(outShapeInfo2, (INPUT_VARIABLE(2))->ordering());
         
    return new ShapeList({outShapeInfo1, outShapeInfo2});
}   








}
}


