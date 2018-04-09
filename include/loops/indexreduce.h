/*
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include "../helpers/shape.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dll.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include <helpers/TAD.h>


#include "../pairwise_util.h"

#include "legacy_ops.h"

namespace functions {
	namespace indexreduce {

		template<typename T>
		class IndexReduce {
		public:
#ifdef __CUDACC__

		static __device__ void transform(const int opNum, T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfo, int *dimension,int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffset);

			/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
    template<typename OpType>
	static __device__ void aggregatePartials(IndexValue<T> **sPartialsRef,int tid,int numElements,T *extraParams);

	/**
	 * @param n n is the number of
	 *        elements to loop through
	 * @param dx the data to operate on
	 * @param xVectorInfo the meta data for the vector:
	 *                              0 is the offset
	 *                              1 is the increment/stride
	 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
	 *                              3 is the element wise stride for the buffer
	 *                              4 is the number of elements it takes to get to the next row/column/tensor
	 * @param gpuInformation
	 *                              0 is the block size
	 *                              1 is the grid size
	 *                              2 is the shared memory size
	 * @param problemDefinition
	 *                          0 is the number of elements per vector
	 *                          1 is the number of vectors
	 */
    template<typename OpType>
	static __device__ void transform(T *dx, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets);

    template <typename T>
    __device__ void indexReduceGeneric(const int op, T *dx, int *xShapeInfo, int xRank, T *extraParams, T *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets);
#endif
		static T execScalar(const int opNum, T *x, int *xShapeInfo, T *extraParams);

		static void exec(const int opNum, T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffset);

		template<typename OpType>
		static _CUDA_H T execScalar(T *x, int *xShapeInfo, T *extraParams);

		template<typename OpType>
		static _CUDA_H void exec(T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffset);
		};
	}
}


#ifdef __CUDACC__


__global__ void indexReduceDouble(
        int op,
        double *dx,
        int *xShapeInfo, int xRank,
        double *extraParams,
        double *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationBuffer, double *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<double>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void indexReduceFloat(
        int op,
        float *dx,
        int *xShapeInfo, int xRank,
        float *extraParams,
        float *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot,  int *allocationBuffer, float *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<float>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}

__global__ void indexReduceHalf(
        int op,
        float16 *dx,
        int *xShapeInfo, int xRank,
        float16 *extraParams,
        float16 *result,
        int *resultShapeInfo, int zRank,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot,  int *allocationBuffer, float16 *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {
    indexReduceGeneric<float16>(
            op,
            dx,
            xShapeInfo, xRank,
            extraParams,
            result,
            resultShapeInfo, zRank,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

}


#endif

#endif /* INDEXREDUCE_H_ */

