//
// Created by yurii@skymind.io on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <vector>
#include <numeric>

namespace nd4j {
    namespace ops {

// declare auxiliary function
template<typename T>
void recursiveLoop(Block<T>& block, NDArray<T>* input, const NDArray<T>* paddings, NDArray<T>* output, std::vector<int>& dimensions, int dim);


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(pad, 2, 1, false, 0, 1) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    NDArray<T>* output   = OUTPUT_VARIABLE(0);
    std::vector<int>* argI = block.getIArguments();

	std::vector<int> dimensions(input->rankOf());	
    std::iota(dimensions.begin(), dimensions.end(), 0);   			// fill with 0, 1, ... rank-1
    int dim = 0;

	switch(argI->at(0)) {
		case 0:				// CONSTANT mode
			recursiveLoop(block, input, paddings, output, dimensions, dim);
			break;
		case 1:				// REFLECT mode
			break;
		case 2:				// SYMMETRIC mode
			break;
		default:
			throw "CUSTOM_OP pad: unknown padding mode, there are only three possible legal values - 0,1,2 !";
	}

    STORE_RESULT(*output);
	
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(pad) {

	// check shape of paddings 
	NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    int rank =  input->rankOf();    

	if (paddings->rankOf() != 2 || paddings->shapeOf()[0] != rank || paddings->shapeOf()[1] != 2)
		throw "CUSTOM_OP pad: wrong shape of input paddings !";

	std::vector<int>* argI = block.getIArguments();

	// in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements 
	// REFLECT case
	if(argI->at(0) == 1)				
		for(int dim=0; dim < rank; ++dim)
			if(!(paddings->getScalar(dim,0) <= (input->shapeOf()[dim]-1) && paddings->getScalar(dim,1) <= (input->shapeOf()[dim]-1)))
				throw "CUSTOM_OP pad: wrong shape of input paddings for REFLECT mode !";
	// REFLECT case
	if(argI->at(0) == 2)				
	for(int dim=0; dim < rank; ++dim)
		if(!(paddings->getScalar(dim,0) <= input->shapeOf()[dim] && paddings->getScalar(dim,1) <= input->shapeOf()[dim]))
			throw "CUSTOM_OP pad: wrong shape of input paddings for SYMMETRIC mode !";

	
	int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), rank*2+4, int);
    outShapeInfo[0] = rank;
    for(int i=1; i <= rank; ++i)
    	outShapeInfo[i] = input->shapeOf()[i-1] + paddings->getScalar(i-1,0) + paddings->getScalar(i-1,1);
	
    shape::updateStrides(outShapeInfo, input->ordering());    

    return new ShapeList(outShapeInfo);
    
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void recursiveLoop(Block<T>& block, NDArray<T>* input, const NDArray<T>* paddings, NDArray<T>* output, std::vector<int>& dimensions, int dim ) {   // initial values of inIdx,outIdx,dim have to be zero
	
	dimensions.erase(dimensions.begin());	
	shape::TAD tadOut(output->getShapeInfo(), dimensions.data(), dimensions.size());
    tadOut.createTadOnlyShapeInfo();
    tadOut.createOffsets();
    NDArray<T> subArrOut(output->getBuffer(), tadOut.tadOnlyShapeInfo, block.getWorkspace());
    
    shape::TAD* tadIn = nullptr;
    NDArray<T>* subArrIn = nullptr;
    if(dim == input->rankOf()-2) {
    	tadIn = new  shape::TAD(input->getShapeInfo(), dimensions.data(), dimensions.size());
    	tadIn->createTadOnlyShapeInfo();
    	tadIn->createOffsets();
    	subArrIn = new NDArray<T>(input->getBuffer(), tadIn->tadOnlyShapeInfo, block.getWorkspace());
    }

    for(int i = 0; i < output->shapeOf()[dim]; ++i) {		
		subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[i]);	
		if(i < (int)paddings->getScalar(dim,0) || i >= (input->shapeOf()[dim] + (int)paddings->getScalar(dim,0))) 			// corresponds to outer range (relevant indexes are absent in input)						
			subArrOut.assign((T)0.);					
		else {			
			if(dim < input->rankOf()-2)
				recursiveLoop(block, input, paddings, output, dimensions, ++dim);
			else {		// now we are on next to last dim = rank-2								
    			subArrIn->setBuffer(input->getBuffer() + tadIn->tadOffsets[i - (int)paddings->getScalar(dim,0)]);		    			
				// most inner loop, corresponds to last dim = rank-1
				for(int j=0; j < output->shapeOf()[dim+1]; ++j) 					
					if(j < (int)paddings->getScalar(dim+1,0) || j >= (input->shapeOf()[dim+1] + (int)paddings->getScalar(dim+1,0))) 
						subArrOut.putIndexedScalar(j, (T)0.);											
					else 															
						subArrOut.putIndexedScalar(j, subArrIn->getIndexedScalar(j - (int)paddings->getScalar(dim+1,0)));					
			}
		}
	}
	delete tadIn;
	delete subArrIn;			
}





}
}