//
//  @author raver119@gmail.com
//

#include <Environment.h>
#include <loops/transform.h>
#include <op_boilerplate.h>

#include <loops/legacy_ops.h>


namespace functions {
    namespace transform {

        template <typename T>
        template <typename OpType>
        __device__ void Transform<T>::transformCuda(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

		    if(OpType::requiresSpecial) {
			    OpType::execSpecialCuda(dy,shapeInfo,result,resultShapeInfo,params, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			    return;
		    } else {

    		    int *xShape = shape::shapeOf(shapeInfo);
	    	    int *xStride = shape::stride(shapeInfo);
		        char xOrder = shape::order(shapeInfo);
		        char resultOrder = shape::order(resultShapeInfo);
    		    int xRank = shape::rank(shapeInfo);
	    	    int xOffset = shape::offset(shapeInfo);

		        int xElementWiseStride = shape::elementWiseStride(shapeInfo);
    		    int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
	    	    int tid = blockIdx.x * blockDim.x + threadIdx.x;

                __shared__ int length;
		        if(threadIdx.x == 0)
			        length = shape::length(shapeInfo);
		        __syncthreads();

		        if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == resultOrder) {
			        transformCuda<OpType>(
				    	length,
				    	dy,
				    	xElementWiseStride,
				    	params,
				    	result,
				    	resultElementWiseStride, allocationPointer, reductionPointer, manager);
		        }
		        else {
			        int xCoord[MAX_RANK];
			
		    	    for (Nd4jIndex i = tid; i < length; i+= gridDim.x * blockDim.x) {
			    	    shape::ind2sub(xRank,shape::shapeOf(shapeInfo),i, xCoord);
				        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
    				    Nd4jIndex resultOffset2 = shape::getOffset(0,xShape,shape::stride(resultShapeInfo),xCoord,xRank);
	    			    result[resultOffset2] = OpType::op(dy[xOffset2], params);
		    	    }
		        }
	        }
	    };

        template <typename T>
        template <typename OpType>
	    __device__ void Transform<T>::transformCuda(
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
		
            int totalThreads = gridDim.x * blockDim.x;
		    Nd4jIndex i = blockIdx.x * blockDim.x + threadIdx.x;

    		if(incy == 1 && resultStride == 1) {
	    		/* equal, positive, non-unit increments. */
			    for (; i < n; i += totalThreads) {
				    result[i] = OpType::op(dy[i], params);
			    }
		    }
		    else {
			    for (; i < n; i += totalThreads) {
				    result[i * resultStride] = OpType::op(dy[i * incy], params);
			    }
		    }
	    }


        template <typename T>
        __device__ void Transform<T>::transformCuda(
			const int opNum,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                DISPATCH_BY_OPNUM(transformCuda, PARAMS(dy, shapeInfo, params, result, resultShapeInfo, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
	    }

        template <typename T>
        __device__ void Transform<T>::transformCuda(
			const int opNum,
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager) {
                                DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dy, incy, params, result, resultStride, allocationPointer, reductionPointer, manager), TRANSFORM_OPS);
	    }
    }
}