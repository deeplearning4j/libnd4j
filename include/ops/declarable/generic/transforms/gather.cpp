//
// @author Shyrma Yurii (iuriish@yahoo.com), created on 16.11.2017
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather, 1, 1, false, 0, -2) {

	NDArray<T>* input   = INPUT_VARIABLE(0);
    NDArray<T>* indices = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
	NDArray<T>* output  = OUTPUT_VARIABLE(0);

	const int numOfIntArgs = block.numI();

    std::vector<int> intArgs;
    if(numOfIntArgs == 0)
        intArgs.emplace_back(0);
    else
        for(int i=0; i<numOfIntArgs; ++i)
            intArgs.emplace_back(block.getIArguments()->at(i));

    const int inputRank = input->rankOf();
	if(intArgs[0] < 0)
        intArgs[0] += inputRank;

	// input validation
    REQUIRE_TRUE(intArgs[0] < inputRank, 0, "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!", intArgs[0], inputRank);
    REQUIRE_TRUE(indices || numOfIntArgs > 1, 0, "GATHER op: indices should be provided either as additional input array or as IntArguments !");

	helpers::gather(input, indices, output, intArgs);

    return Status::OK();
}


DECLARE_SHAPE_FN(gather) {

	// check shape of paddings 
	int* inputShapeInfo  = inputShape->at(0);
	int* outputShapeInfo = nullptr;

	int axis = block.numI() > 0 ? block.getIArguments()->at(0) : 0;
	int inputRank = inputShapeInfo[0];
	if(axis < 0)
		axis += inputRank;

    REQUIRE_TRUE(axis < inputRank, 0, "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!", axis, inputRank);

	if (block.width() > 1) {
		int* indicesShapeInfo = inputShape->at(1);
    
    	int indicesRank = indicesShapeInfo[0];
        
    	if(shape::isScalar(indicesShapeInfo))
    		indicesRank = 0;
    	else if(shape::isVector(indicesShapeInfo))
    		indicesRank = 1;

    	int outputRank = inputRank + indicesRank - 1;
    	ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), int);
    	// fill output shapeInfo
    	outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;    
    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];
    	
		if(!shape::isScalar(indicesShapeInfo) && shape::isVector(indicesShapeInfo))
    		outputShapeInfo[shapeIdx++] = shape::length(indicesShapeInfo);
    	else if(!shape::isScalar(indicesShapeInfo))
    		for(int i = 0; i < indicesShapeInfo[0]; ++i)
    			outputShapeInfo[shapeIdx++] = indicesShapeInfo[i+1];

    	for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];
	
    	shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));    
	} else if (block.numI() > 1) {
		int indicesRank = block.numI() == 2 ? 0 : 1;

		int outputRank = inputRank + indicesRank - 1;
		ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), int);

		// building shape manually
		outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;    
    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];

		if (block.numI() > 2)
			outputShapeInfo[shapeIdx++] = block.numI() - 1;

		for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];

		shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));    
	}
    else
        REQUIRE_TRUE(false, 0, "GATHER op: indices should be provided either as additional input array or as IntArguments !");


    return SHAPELIST(outputShapeInfo);
    
}





}
}
