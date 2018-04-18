//
// Created by farizrahman4u@gmail.com on 02.04.2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/stack.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(parallel_concat, -1, 1, false, 0, 0) {
	
	NDArray<T>* input  = INPUT_VARIABLE(0);
	NDArray<T>* output = OUTPUT_VARIABLE(0);

	// check whether shapes of all input array are the same		
	for (int i = 0; i < (int) block.width() - 1; ++i) {
		REQUIRE_TRUE(shape::equalsSoft((INPUT_VARIABLE(i))->getShapeInfo(), (INPUT_VARIABLE(i+1))->getShapeInfo()), 0, "CUSTOM_OP parallel_concat: the shapes of input arrays must be the same !");
 	 }

 	// check whether first dimension is 1
 	 REQUIRE_TRUE(input->getShapeInfo()[1] == 1, 0, "CUSTOM_OP parallel_concat: first dimension of input arrays must be 1 !");

 	std::vector<NDArray<T>*> inArrs(block.width());
 	for(int i = 0; i < block.width(); ++i)
		inArrs[i] = INPUT_VARIABLE(i);
	
	const int dim = 0;
	helpers::stack(inArrs, *output, dim);
	 	
  	return Status::OK();
}


DECLARE_SHAPE_FN(parallel_concat) {
	
	int* inShapeInfo = inputShape->at(0);
	int rank = inShapeInfo[0];

	int* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);

	outShapeInfo[0] = rank;
	outShapeInfo[1] = block.width();
	for (int i = 2; i <= rank; ++i){
		outShapeInfo[i] = inShapeInfo[i];
	}
	shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));
  	
  	return SHAPELIST(outShapeInfo);
}


}
}