//
//  @author raver119@gmail.com
//

#include <Environment.h>
#include <loops/transform.h>
#include <op_boilerplate.h>

#include <loops/legacy_ops.h>




template <typename T>
__device__ void transformGeneric(
		int opNum,
		Nd4jIndex n,
		T *dy,
		int incy,
		T *params,
		T *result,
		int resultStride, int *allocationPointer, T *reductionPointer) {

	functions::transform::Transform<T>::transformCuda(
		opNum,
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		nullptr);
}

template <typename T, typename OpClass>
__device__ void transformSimpleGeneric(
		Nd4jIndex n,
		T *dy,
		int incy,
		T *params,
		T *result,
		int resultStride, int *allocationPointer, T *reductionPointer) {

	functions::transform::Transform<T>::template transformCuda<OpClass>(
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		nullptr);
}



template <typename T>
__device__ void transformGeneric(
		int opNum,
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer) {

	functions::transform::Transform<T>::transformCuda(
	    opNum,
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    nullptr);
}

template <typename T, typename OpClass>
__device__ void transformSimpleGeneric(
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
	
    functions::transform::Transform<T>::template transformCuda<OpClass>(
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    nullptr, tadShapeInfo, tadOffsets);
}

// transform strided
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float, INPUT(Nd4jIndex n, float *x, int xStride, float *extraParams, float *z, int zStride, int *allocationPointer, float *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, double, INPUT(Nd4jIndex n, double *x, int xStride, double *extraParams, double *z, int zStride, int *allocationPointer, double *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float16, INPUT(Nd4jIndex n, float16 *x, int xStride, float16 *extraParams, float16 *z, int zStride, int *allocationPointer, float16 *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

// transform shaped
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float, INPUT(float *x, int *xShape, int xRank, float *extraParams, float *z, int *zShape, int zRank, int *allocationPointer, float *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, double, INPUT(double *x, int *xShape, int xRank, double *extraParams, double *z, int *zShape, int zRank, int *allocationPointer, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float16, INPUT(float16 *x, int *xShape, int xRank, float16 *extraParams, float16 *z, int *zShape, int zRank, int *allocationPointer, float16 *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))



namespace functions {
    namespace transform {

        template <>
        _CUDA_H void Transform<float>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jIndex n, float *x, int xStride, float *extraParams, float *z, int zStride, int *allocationPointer, float *reductionPointer) {
            DISPATCH_SIMPLE(transformStrided, float, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	        if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        };

        template <>
        _CUDA_H void Transform<double>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jIndex n, double *x, int xStride, double *extraParams, double *z, int zStride, int *allocationPointer, double *reductionPointer) {
            DISPATCH_SIMPLE(transformStrided, double, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	        if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        };

        template <>
        _CUDA_H void Transform<float16>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jIndex n, float16 *x, int xStride, float16 *extraParams, float16 *z, int zStride, int *allocationPointer, float16 *reductionPointer) {
            DISPATCH_SIMPLE(transformStrided, float16, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

	        if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        };

        template <>
        _CUDA_H void Transform<float>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, float *x, int *xShape, int xRank, float *extraParams, float *z, int *zShape, int zRank, int *allocationPointer, float *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets) {
            
            DISPATCH_SIMPLE(transformShaped, float, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
            
            if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        }

        template <>
        _CUDA_H void Transform<float16>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, float16 *x, int *xShape, int xRank, float16 *extraParams, float16 *z, int *zShape, int zRank, int *allocationPointer, float16 *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets) {
            
            DISPATCH_SIMPLE(transformShaped, float16, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
            
            if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        }

        template <>
        _CUDA_H void Transform<double>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, double *x, int *xShape, int xRank, double *extraParams, double *z, int *zShape, int zRank, int *allocationPointer, double *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets) {
            
            DISPATCH_SIMPLE(transformShaped, double, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
            
            if (nd4j::Environment::getInstance()->isDebug())
		        checkCudaErrors(cudaStreamSynchronize(*stream));
        }

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


        //template class ND4J_EXPORT Transform<float>;
        //template class ND4J_EXPORT Transform<float16>;
        //template class ND4J_EXPORT Transform<double>;

        BUILD_CALL_1(template __device__ void Transform<float>::transformCuda, float, (float*, int*, float*, float*,int*, int*,float*, UnifiedSharedMemory*, int*, Nd4jIndex*), TRANSFORM_OPS)
        BUILD_CALL_1(template __device__ void Transform<float16>::transformCuda, float16, (float16*, int*, float16*, float16*,int*, int*, float16*, UnifiedSharedMemory*, int*, Nd4jIndex*), TRANSFORM_OPS)
        BUILD_CALL_1(template __device__ void Transform<double>::transformCuda, double, (double*, int*, double*, double*,int*, int*, double*, UnifiedSharedMemory*, int*, Nd4jIndex*), TRANSFORM_OPS)
    }
}
