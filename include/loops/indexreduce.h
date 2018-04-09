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

#ifdef __CUDACC__
		// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template<typename T>
struct SharedIndexValue {
	// Ensure that we won't compile any un-specialized types
	__device__ T * getPointer() {
		extern __device__ void error(void);
		error();
		return 0;
	}
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<float> {
	__device__ IndexValue<float> * getPointer() {
		extern __shared__ IndexValue<float> s_int2[];
		return s_int2;
	}
};
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template<>
struct SharedIndexValue<double> {
	__device__ IndexValue<double> * getPointer() {
		extern __shared__ IndexValue<double> s_int6[];
		return s_int6;
	}
};
#endif

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
	static __device__ void transform(T *dx, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, sNd4jIndex *tadOffsets);

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


#endif

#endif /* INDEXREDUCE_H_ */

