/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *  @author: agibsonccc
 *  @author: raver119@gmail.com
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <vector>
#include <templatemath.h>
#include <op.h>
#include <omp.h>
#include <pairwise_util.h>
#include <dll.h>
#include "reduce.h"
#include "scalar.h"
#include "indexreduce.h"
#include "broadcasting.h"
#include <shape.h>
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif

#ifdef __JNI__
#include <jni.h>
#endif
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif


namespace functions {
    namespace transform {

        template<typename T>
        class Transform : public functions::ops::Op<T> {
        protected:
            bool requiresSpecial = false;

        public:

            /**
             *
             */
            virtual void execSpecial(
                    T *dx,
                    int *xShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams) = 0;

#ifdef __CUDACC__
            /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) = 0;
#endif
            /**
             * The op for transforms
             * @param d1
             * @param params
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __device__ __host__

#elif defined(__GNUC__)


#endif
            T op(T d1, T *params) = 0;

#ifdef __CUDACC__
            /**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transform(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes) {
		Nd4jIndex n = shape::length(shapeInfo);
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + tid;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = op(dy[indexes[i]], params);
		}

	}


	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transformCuda(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {

		if(this->requiresSpecial) {
			this->execSpecialCuda(dy,shapeInfo,result,resultShapeInfo,params, allocationPointer, reductionPointer, manager);
			return;
		}

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
			transformCuda(
					length,
					dy,
					xElementWiseStride,
					params,
					result,
					resultElementWiseStride, allocationPointer, reductionPointer, manager);
		}
		else {
			/* equal, positive, non-unit increments. */
			//long allocSize = sizeof(int) * xRank;
			//int *xIdx = shape::cuMalloc(manager->getT1ShapeBuffer(), allocSize);
			int xCoord[MAX_RANK];

#pragma unroll
			for (int i = tid; i < length; i+= gridDim.x * blockDim.x) {
				//int *xIdx = shape::ind2sub(xRank, xShape, i, xIdx);
				shape::ind2sub(xRank,shape::shapeOf(shapeInfo),i, xCoord);
				Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
				Nd4jIndex resultOffset2 = shape::getOffset(0,xShape,shape::stride(resultShapeInfo),xCoord,xRank);
				result[resultOffset2] = op(dy[xOffset2], params);
			}
		}
	}

	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual  __inline__ __device__ void transformCuda(
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {
		int totalThreads = gridDim.x * blockDim.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + threadIdx.x;

		if(incy == 1 && resultStride == 1) {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i] = op(dy[i], params);
			}
		}
		else {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * resultStride] = op(dy[i * incy], params);
			}
		}


	}
#endif

            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams,
                    Nd4jIndex *indexes) {
                int n = shape::length(xShapeInfo);
#pragma omp simd
                for (int i = 0; i < n; i++) {
                    result[indexes[i]] = op(dx[indexes[i]], extraParams);
                }
            }

            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams,
                     Nd4jIndex *indexes,
                    Nd4jIndex *resultIndexes) {
                int n = shape::length(xShapeInfo);
#pragma omp parallel for
                for (int i = 0; i < n; i++) {
                    result[resultIndexes[i]] = op(dx[indexes[i]], extraParams);
                }
            }


            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams) {

                if(this->requiresSpecial) {
                    this->execSpecial(dx,xShapeInfo,result,resultShapeInfo,extraParams);
                    return;
                }

                int n = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo)) {
                    exec(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    int *resultStride = shape::stride(resultShapeInfo);
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 dx,
                                                 xStride,
                                                 result,
                                                 resultStride,
                                                 &rank,
                                                 shapeIter,
                                                 &dx,
                                                 xStridesIter,
                                                 &result,
                                                 resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
                        {
                            /* Process the innermost dimension */
                            T *xIter = dx;
                            T *resultIter = result;
                            resultIter[0] = op(xIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               dx,
                                               xStridesIter,
                                               result,
                                               resultStridesIter);

                    }

                }

            }


            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(T *dx,
                              int xStride,
                              T *result,
                              int resultStride,
                              T *extraParams,
                              int n) {
                if (xStride == 1 && resultStride == 1) {
                    if(n < 8000) {
#pragma omp simd
                        for (int i = 0; i < n; i++) {
                            result[i] = op(dx[i], extraParams);
                        }
                    }
                    else {
#pragma omp parallel  for
                        for (int i = 0; i < n; i++) {
                            result[i] = op(dx[i], extraParams);
                        }
                    }

                }


                else {
                    if(n < 8000) {
#pragma omp simd
                        for (int i = 0; i < n; i++) {
                            result[i * resultStride] = op(dx[i * xStride],
                                                          extraParams);
                        }
                    }
                    else {
#pragma omp parallel for
                        for (int i = 0; i < n; i++) {
                            result[i * resultStride] = op(dx[i * xStride],
                                                          extraParams);
                        }
                    }

                }

            }
            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                //no op aggregation needs to happen for transforms
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)


#endif
            virtual ~Transform() {
            }
#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)


#endif
            Transform() {
            }


        };

        namespace ops {
/**
 * abs(x)
 */
            template<typename T>
            class Abs : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_abs<T>(d1);
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Abs() {
                }

            };

/**
 * cei(x)
 */
            template<typename T>
            class Ceiling : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_ceil<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Ceiling() {
                }

            };

/**
 * cos(x)
 */
            template<typename T>
            class Cosine : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_cos<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Cosine() {
                }

            };

/**
 * exp(x)
 */
            template<typename T>
            class Exp : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_exp<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Exp() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class HardTanhDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return ((d1 >= -1.0 && d1 <= 1.0) ? 1.0 : 0.0);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~HardTanhDerivative() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class HardTanh : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return d1 < -1.0 ? -1.0 : d1 > 1.0 ? 1.0 : d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~HardTanh() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class Floor : public Transform<T> {

            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_floor<T>(d1);
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Floor() {
                }

            };

/**
 * log(x)
 */
            template<typename T>
            class Log : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_log<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Log() {
                }

            };

            template<typename T>
            class SpecialDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return d1 * (1.0 - d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SpecialDerivative() {
                }

            };

/**
 * -x
 */
            template<typename T>
            class Neg : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return -d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Neg() {
                }

            };

/**
 * pow(x,extra params [0])
 */
            template<typename T>
            class Pow : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_pow<T>(d1, params[0]);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Pow() {
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                Pow() {
                }
            };

/**
 * round(x)
 */
            template<typename T>
            class Round : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_round<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Round() {
                }

            };

/**
 * sigmoid(x)
 */
            template<typename T>
            class Sigmoid : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sigmoid<T>(d1);
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Sigmoid() {
                }

            };


/**
 * sigmoid(x)
 */
            template<typename T>
            class SigmoidDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sigmoidderivative<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SigmoidDerivative() {
                }

            };


/**
 * Scale to be between a
 * min and max
 */
            template<typename T>
            class SetRange : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    T min = params[0];
                    T max = params[1];
                    if (d1 >= min && d1 <= max)
                        return d1;
                    if (min == 0 && max == 1) {
                        T val = 1 / (1 + nd4j::math::nd4j_exp<T>(-d1));
                        return (nd4j::math::nd4j_floor<T>(val * (max - min)) + min);
                    }

                    T ret = (nd4j::math::nd4j_floor<T>(d1 * (max - min)) + min);
                    return ret;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SetRange() {
                }

            };

/**
 * sin(x)
 */
            template<typename T>
            class Sin : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sin<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Sin() {
                }

            };

/**
 * sqrt(x)
 */
            template<typename T>
            class Sqrt : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sqrt<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Sqrt() {
                }

            };

/**
 * softplus(x)
 */
            template<typename T>
            class SoftPlus : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SoftPlus() {
                }


            };

/**
 * sign(x)
 */
            template<typename T>
            class Sign : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return (d1 > 0) - (d1 < 0);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Sign() {
                }
            };


/**
 * tanh(x)
 */
            template<typename T>
            class TimesOneMinus : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return d1 * (1 - d1);
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~TimesOneMinus() {
                }

            };


/**
 * tanh(x)
 */
            template<typename T>
            class Tanh : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_tanh<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Tanh() {
                }

            };

/**
 * tanh(x)
 */
            template<typename T>
            class TanhDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_tanhderivative<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~TanhDerivative() {
                }

            };

/**
 * acos(x)
 */
            template<typename T>
            class ACos : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_acos<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~ACos() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class Ones : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return 1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Ones() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class SoftSign : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return nd4j::math::nd4j_softsign<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SoftSign() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class SoftSignDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return nd4j::math::nd4j_softsignderivative<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SoftSignDerivative() {
                }

            };

/**
 * asin(x)
 */
            template<typename T>
            class ELU : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_elu<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~ELU() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class ELUDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_eluderivative<T>(d1);
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~ELUDerivative() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class RELU : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return d1 < params[0] ? params[0] : d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~RELU() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class LeakyRELU : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_leakyrelu<T>(d1, params[0]);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~LeakyRELU() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class LeakyRELUDerivative : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return (d1 >= 0 ? 1.0 : params[0]);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~LeakyRELUDerivative() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class ASin : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_asin<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~ASin() {
                }

            };

/**
 * atan(x)
 */
            template<typename T>
            class ATan : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_atan(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~ATan() {
                }

            };

/**
 * atan(x)
 */
            template<typename T>
            class Identity : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Identity() {
                }

            };

/**
 * atan(x)
 */
            template<typename T>
            class Stabilize : public Transform<T> {
            public:
                double realMin = 1.1755e-38f;
                double cutOff = nd4j::math::nd4j_log(realMin);

                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    T k = params[0];
                    if (d1 * k > -cutOff)
                        return (float) (-cutOff / k);
                    else if (d1 * k < cutOff)
                        return (float) (cutOff / k);
                    return d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Stabilize() {
                }

            };


/**
 * atan(x)
 */
            template<typename T>
            class Step : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif


                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return (d1 > params[0] ? 1.0 : 0.0);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Step() {
                }

            };


/**
 * atan(x)
 */
            template<typename T>
            class OneMinus : public Transform<T> {
            public:
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {//no-op
                }

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {}
#endif

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return 1.0 - d1;
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~OneMinus() {
                }

            };


            template<typename T>
            class Im2col : public Transform<T> {
            public:

                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                int outSize(int size, int k, int s, int p, bool coverAll) {
                    if (coverAll)
                        return (size + p * 2 - k + s - 1) / s + 1;
                    else
                        return (size + p * 2 - k) / s + 1;
                }

#ifdef __CUDACC__
                /**
	 * Based on:  https://github.com/pjreddie/darknet/blob/master/src/im2col_kernels.cu
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {
		/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
		int kernelWidth = (int) extraParams[0];
		int kernelHeight = (int) extraParams[1];
		int strideX = (int) extraParams[2];
		int strideY = (int) extraParams[3];
		int padWidth = (int) extraParams[4];
		int padHeight = (int) extraParams[5];
		int kSize = kernelWidth * kernelHeight;

		int *outShape = shape::shapeOf(resultShapeBuffer);
		char resultOrder = shape::order(resultShapeBuffer);
		int *outStride = shape::stride(resultShapeBuffer);

		int *inShape = shape::shapeOf(xShapeBuffer);
		int *inStride = shape::stride(xShapeBuffer);

		int samples = inShape[0];
		int depth = inShape[1];
		int height = inShape[2];
		int width = inShape[3];


        int strideex = inStride[0];
        int stridech = inStride[1];
        int strideh = inStride[2];
        int stridew = inStride[3];

		// (height + 2 * padHeight - kernelHeight) / strideX + 1; //
		// (width + 2 * padWidth - kernelWidth) / strideY + 1; //
		int height_col = outShape[4];
		int width_col =  outShape[5];

		int n = samples * depth * height_col * width_col;
/*
		if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, height, width, depth, n, samples);
*/

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		for(; index < n; index += blockDim.x*gridDim.x) {
			int h_index = index / width_col;
			int h_col = h_index % height_col;
			int w_col = index % width_col;

			int c_im = h_index / height_col;
			int c_col = c_im * kSize;

            int depth_im = c_im % depth;
            int num_im = c_im / depth;
			int h_offset = h_col * strideY - padHeight;
			int w_offset = w_col * strideX - padWidth;

			T* data_col_ptr = result;

			int i_c = (c_col * height_col + h_col) * width_col + w_col;
			data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

			 T* data_im_ptr = dx;

            data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

			for (int i = 0; i < kernelHeight; ++i) {
				for (int j = 0; j < kernelWidth; ++j) {
					int h_im = h_offset + i;
					int w_im = w_offset + j;
                        if (resultOrder == 'c') {
                    		*data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                        }
                        else {
                        	int i_f = 0;
							int i_c_temp = i_c;
                        	for (int dim=5;dim >= 0;dim--)
                                {
                                        i_f += ( i_c_temp % outShape[dim] )  * outStride[dim];
                                        i_c_temp = i_c_temp / outShape[dim];
                                }
							result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                        }
						data_col_ptr += height_col * width_col;
						i_c += height_col * width_col;
				}
			}
		}
	}
#endif

                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    /*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
                    int kernelWidth = (int) extraParams[0];
                    int kernelHeight = (int) extraParams[1];
                    int strideX = (int) extraParams[2];
                    int strideY = (int) extraParams[3];
                    int padWidth = (int) extraParams[4];
                    int padHeight = (int) extraParams[5];
                    bool coverAll = extraParams[6] > 0.0;

                    int outArrayOffset = 0;
                    int *outShape = shape::shapeOf(resultShapeBuffer);
                    int *outStride = shape::stride(resultShapeBuffer);

                    int inArrayOffset = 0;
                    int *inShape = shape::shapeOf(xShapeBuffer);
                    int *inStride = shape::stride(xShapeBuffer);


                    int exampleFrom = 0;
                    int exampleTo = inShape[0];
                    int depthFrom = 0;
                    int depthTo = inShape[1];
                    int yOutFrom = 0;
                    int yOutTo = this->outSize(inShape[2], kernelHeight, strideY, padHeight, coverAll);
                    int xOutFrom = 0;
                    int xOutTo = this->outSize(inShape[3], kernelWidth, strideX, padWidth, coverAll);


                    int *outIndices = new int[6];
                    int *inIndices = new int[4];

                    int inStride2 = inStride[2];
                    int inStride3 = inStride[3];
                    int outStride2 = outStride[2];
                    int outStride3 = outStride[3];
                    int inShape2 = inShape[2];
                    int inShape3 = inShape[3];

                     bool padding = padHeight > 0 || padWidth > 0;

                    T *dIn = dx;
                    T *dOut = result;
                    //#pragma omp parallel for collapse(2)
                    for (int ex = exampleFrom; ex < exampleTo; ex++) {
                        for (int d = depthFrom; d < depthTo; d++) {
                            inIndices[0] = ex;
                            inIndices[1] = d;
                            outIndices[0] = ex;
                            outIndices[1] = d;

                            for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                                for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                                    outIndices[4] = y;
                                    outIndices[5] = x;
                                    int baseOffsetOut = this->getOffsetUnsafe6(outArrayOffset, outShape, outStride,
                                                                               outIndices);

                                    if (padding) {
                                        int i = y * strideY -
                                                padHeight;    //index along height of first element of patch in original img
                                        int j = x * strideX -
                                                padWidth;     //index along width of first element in patch in original img
                                        inIndices[2] = i;   //along height
                                        inIndices[3] = j;   //along width

                                        int baseOffsetIn = this->getOffsetUnsafe4(inArrayOffset, inShape, inStride,
                                                                                  inIndices);
                                        if (outStride2 <= outStride3) {
                                            //Want dimension 2 (along height) in inner loop for cache reasons
                                            for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                                int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                    if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 ||
                                                        j + patchX >= inShape3)
                                                        dOut[outBufferIdxX + patchY * outStride2] = 0; //padding
                                                    else {
                                                        dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
                                                                                                        patchY *
                                                                                                        inStride2];
                                                    }
                                                }
                                            }
                                        } else {
                                            //Want dimension 3 in inner loop for cache reasons
                                            for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                                int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                    if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] ||
                                                        j + patchX >= inShape[3])
                                                        dOut[outBufferIdxY + patchX * outStride3] = 0.0; //padding
                                                    else {
                                                        dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
                                                                                                        patchX *
                                                                                                        inStride3];
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        //No padding
                                        int i = y *
                                                strideY;    //index along height of first element of patch in original img
                                        int j = x *
                                                strideX;     //index along width of first element in patch in original img
                                        inIndices[2] = i;   //along height
                                        inIndices[3] = j;   //along width

                                        int baseOffsetIn = this->getOffsetUnsafe4(inArrayOffset, inShape, inStride,
                                                                                  inIndices);
                                        if (outStride2 <= outStride3) {
                                            //Want dimension 2 (along height) in inner loop for cache reasons
                                            for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                                int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                    dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
                                                                                                    patchY * inStride2];
                                                }
                                            }
                                        } else {
                                            //Want dimension 3 in inner loop for cache reasons
                                            for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                                int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                    dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
                                                                                                    patchX * inStride3];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    delete[] inIndices;
                    delete[] outIndices;

                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Im2col() {
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                Im2col() {
                    this->requiresSpecial = true;
                }


                /** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
                 *  normally negative indices are bad, OK here because of other checks on input indices
                 *  Uses unrolled loop specifically for length 4
                 */
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
                    int offset = baseOffset;
                    if (shape[0] != 1) offset += indices[0] * stride[0];
                    if (shape[1] != 1) offset += indices[1] * stride[1];
                    if (shape[2] != 1) offset += indices[2] * stride[2];
                    if (shape[3] != 1) offset += indices[3] * stride[3];
                    return offset;
                }


                /**
                 * A version of Shape.getOffset without checking on input for negative indices etc
                 * normally negative indices are bad, OK here because of other checks on input indices
                 * Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
                 */
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
                    int offset = baseOffset;
                    if (shape[0] != 1) offset += indices[0] * stride[0];
                    if (shape[1] != 1) offset += indices[1] * stride[1];
                    if (shape[4] != 1) offset += indices[4] * stride[4];
                    if (shape[5] != 1) offset += indices[5] * stride[5];
                    return offset;
                }

            };

            template<typename T>
            class Col2Im : public Transform<T> {

            public:

#ifdef __CUDACC__
                /**
	 * https://github.com/pjreddie/darknet/blob/master/src/col2im_kernels.cu
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {
		int *inShape = shape::shapeOf(xShapeBuffer);
		int *inStride = shape::stride(xShapeBuffer);

        int strideex = inStride[0];
        int stridech= inStride[1];
        int stridekrow = inStride[2];
        int stridekcol = inStride[3];
        int striderow = inStride[4];
        int stridecol = inStride[5];

		int kernelHeight = inShape[2];
		int kernelWidth = inShape[3];

        // C

		int strideX = (int) extraParams[0];
		int strideY = (int) extraParams[1];
		int padWidth= (int) extraParams[2];
		int padHeight = (int) extraParams[3];
		int imgHeight = (int) extraParams[4];
		int imgWidth = (int) extraParams[5];

		int *outShape = shape::shapeOf(resultShapeBuffer);
        char resultOrder = shape::order(resultShapeBuffer);
		int *outStride = shape::stride(resultShapeBuffer);

		int samples = outShape[0];
		int depth = outShape[1];
		//int height = outShape[2];
		//int width = outShape[3];

        int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
    	int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

    	int n = samples * depth * imgHeight * imgWidth;

        /*if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, imgHeight, imgWidth, depth, n, samples);*/



		for(int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
			T val = 0;
			int w_im = i % imgWidth + padWidth;
			int h_im = (i / imgWidth) % imgHeight + padHeight;
			int c_im = i / (imgWidth * imgWidth);

            int num_im = c_im / depth; 
            int depth_im = c_im % depth;

			// compute the start and end of the output
			int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideX + 1;
			int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

			int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideY + 1;
			int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


			for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      			for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        			int h_k = (h_im - h_col * strideY);
        			int w_k = (w_im - w_col * strideX);

	       			int data_col_index =    num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;

			        val += dx[data_col_index];
      			}
		    }
			if (resultOrder == 'c') {
			result[i] += val;
			}
			else {
        		int i_f = 0;
        		int i_c = i;
        		for (int dim=3;dim >= 0;dim--)
            			{
                			i_f += ( i_c % outShape[dim] )  * outStride[dim];
                			i_c = i_c / outShape[dim];
            			}
			result[i_f] += val;
			}
		}
	}
#endif

                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    int inOffset = 0;
                    int *inShape = shape::shapeOf(xShapeBuffer);
                    int *inStride = shape::stride(xShapeBuffer);

                    int kernelHeight = inShape[2];
                    int kernelWidth = inShape[3];
                    /* int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth, */
                    int strideX = (int) extraParams[0];
                    int strideY = (int) extraParams[1];
                    int padWidth = (int) extraParams[2];
                    int padHeight = (int) extraParams[3];


                    int exampleFrom = 0;
                    int exampleTo = inShape[0];
                    int depthFrom = 0;
                    int depthTo = inShape[1];

                    int outArrayOffset = 0;
                    int *outShape = shape::shapeOf(resultShapeBuffer);
                    int *outStride = shape::stride(resultShapeBuffer);


                    int *outIndices = new int[4];
                    int *inIndices = new int[6];

                    int inStride2 = inStride[2];
                    int inStride3 = inStride[3];
                    int outStride2 = outStride[2];
                    int outStride3 = outStride[3];
                    int outShape2 = outShape[2];
                    int outShape3 = outShape[3];

                    int yOutTo = inShape[4];
                    int xOutTo = inShape[5];


                     bool padding = padHeight > 0 || padWidth > 0;

                    T *fIn = dx;
                    T *fOut = result;
                    //#pragma omp parallel for collapse(2)
                    for (int ex = exampleFrom; ex < exampleTo; ex++) {
                        for (int d = depthFrom; d < depthTo; d++) {
                            inIndices[0] = ex;
                            inIndices[1] = d;
                            outIndices[0] = ex;
                            outIndices[1] = d;

                            for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                                for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                                    inIndices[4] = y;   //patch number (along height)
                                    inIndices[5] = x;   //patch number (along width)
                                    int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                                    if (padding) {
                                        int i = y * strideY -
                                                padHeight;    //index along height of first element of patch in original img
                                        int j = x * strideX -
                                                padWidth;     //index along width of first element in patch in original img
                                        outIndices[2] = i;  //along height
                                        outIndices[3] = j;  //along width

                                        int baseOffsetOut = this->getOffsetUnsafe4(outArrayOffset, outShape, outStride,
                                                                                   outIndices);

                                        if (inStride2 <= inStride3) {
                                            //Want dimension 2 (along height) in inner loop for cache efficiency
                                            for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                if (j + patchX < 0 || j + patchX >= outShape3)
                                                    continue;

                                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                    if (i + patchY < 0 || i + patchY >= outShape2)
                                                        continue;
                                                    fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                            fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                                }
                                            }
                                        } else {
                                            //Want dimension 3 (along width) in inner loop for cache efficiency
                                            for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                if (i + patchY < 0 || i + patchY >= outShape2)
                                                    continue;
                                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                    if (j + patchX < 0 || j + patchX >= outShape3)
                                                        continue;
                                                    fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                            fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                                }
                                            }
                                        }
                                    } else {
                                        //No padding
                                        int i = y *
                                                strideY;    //index along height of first element of patch in output img
                                        int j = x *
                                                strideX;     //index along width of first element in patch in output img

                                        outIndices[2] = i;
                                        outIndices[3] = j;

                                        int baseOffsetOut = this->getOffsetUnsafe4(outArrayOffset, outShape, outStride,
                                                                                   outIndices);

                                        if (inStride2 <= inStride3) {
                                            //Want dimension 2 (along height) in inner loop for cache efficiency
                                            for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                    fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                            fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                                }
                                            }
                                        } else {
                                            //Want dimension 3 (along width) in inner loop for cache efficiency
                                            for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                                for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                                    fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                            fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }


                    delete[] outIndices;
                    delete[] inIndices;
                }


                virtual
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                T op(T d1, T *params) {
                    return d1;
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~Col2Im() {
                }

#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                Col2Im() {
                    this->requiresSpecial = true;
                }


                /** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
                 *  normally negative indices are bad, OK here because of other checks on input indices
                 *  Uses unrolled loop specifically for length 4
                 */
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
                    int offset = baseOffset;
                    if (shape[0] != 1) offset += indices[0] * stride[0];
                    if (shape[1] != 1) offset += indices[1] * stride[1];
                    if (shape[2] != 1) offset += indices[2] * stride[2];
                    if (shape[3] != 1) offset += indices[3] * stride[3];
                    return offset;
                }

                /** A version of Shape.getOffset without checking on input for negative indices etc
                 * normally negative indices are bad, OK here because of other checks on input indices
                 * Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
                 */
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
                    int offset = baseOffset;
                    if (shape[0] != 1) offset += indices[0] * stride[0];
                    if (shape[1] != 1) offset += indices[1] * stride[1];
                    if (shape[4] != 1) offset += indices[4] * stride[4];
                    if (shape[5] != 1) offset += indices[5] * stride[5];
                    return offset;
                }

            };


/**
 * softmax(x)
 */
            template<typename T>
            class SoftMax : public Transform<T> {
            public:

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {

		int *shape = shape::shapeOf(xShapeBuffer);
		__shared__ T *maxResult;
		__shared__ int *maxResultShapeBuffer;
		__shared__ functions::reduce::ops::Max<T> *max;
		__shared__ functions::transform::ops::Exp<T> *exp;
		__shared__ functions::broadcast::ops::Subtract<T> *sub;
		__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
		__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
		__shared__ functions::broadcast::ops::Divide<T> *div;
		__shared__ functions::reduce::ops::Sum<T> *sum;
		__shared__ int isVector;

		int length = shape::length(xShapeBuffer);

		if(threadIdx.x == 0) {
			isVector = shape::isVector(xShapeBuffer);
			max = new functions::reduce::ops::Max<T>();
			sum = new functions::reduce::ops::Sum<T>();
			exp = new functions::transform::ops::Exp<T>();
			if (isVector) {
				scalarSub = new functions::scalar::ops::Subtract<T>();
				scalarDiv = new functions::scalar::ops::Divide<T>();
			} else {
				sub = new functions::broadcast::ops::Subtract<T>();
				div = new functions::broadcast::ops::Divide<T>();
			}
			maxResult = new T[shape[0]];
			//printf("maxResult length: [%i]\n", shape[0]);
		}
		__syncthreads();

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int *stride = shape::stride(xShapeBuffer);
		//iterate along rows
		int dimension[1] = {0};
		int maxDimension[1] = {1};
		//compute the row wise maxes

		int maxShape[2] = {shape[0], 1};

		if (threadIdx.x == 0)
			maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);

		if (tid < shape[0])
			maxResult[tid] = (T) 0.0;
		__syncthreads();

		max->transformCuda(dx, xShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension, 1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		if (threadIdx.x == 0) delete max;
		__syncthreads();

		//subtract max of each row
		if (isVector) {
			scalarSub->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		} else {
			sub->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();

		//after subtracting the row wise maxes take the exp
		exp->transformCuda(result, resultShapeBuffer, extraParams,result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
		__syncthreads();


		//take the sum for the exponential
		sum->transformCuda(result, resultShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension,1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		//divide by the sum
		if (isVector) {
			scalarDiv->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		} else {
			div->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();

		if(threadIdx.x == 0) {
			delete sum;
			delete exp;
			if (isVector) {
				delete scalarDiv;
				delete scalarSub;
			} else {
				delete div;
				delete sub;
			}
			delete[] maxResult;
			delete[] maxResultShapeBuffer;
		}
	}
#endif

                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    if (shape::isMatrix(xShapeBuffer)) {
                        int *shape = shape::shapeOf(xShapeBuffer);
                        //iterate along rows
                        int dimension[1] = {0};
                        int maxDimension[1] = {1};
                        //compute the row wise maxes
                        functions::reduce::ops::Max<T> *max = new functions::reduce::ops::Max<T>();
                        std::vector <T> maxResult(shape[0]);
                        for (int i = 0; i < shape[0]; i++)
                            maxResult[i] = 0.0;
                        int maxShape[2] = {shape[0], 1};
                        int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
                        max->exec(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1);

                        //subtract max of each row
                        functions::broadcast::ops::Subtract<T> *sub = new functions::broadcast::ops::Subtract<T>();
                        sub->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);

                        //after subtracting the row wise maxes take the exp
                        functions::transform::ops::Exp<T> *exp = new functions::transform::ops::Exp<T>();
                        exp->exec(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

                        //take the sum for the exponential
                        functions::reduce::ops::Sum<T> *sum = new functions::reduce::ops::Sum<T>();
                        sum->exec(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1);

                        //divide by the sum
                        functions::broadcast::ops::Divide<T> *div = new functions::broadcast::ops::Divide<T>();
                        div->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);


                        delete exp;
                        delete sub;
                        delete sum;
                        delete max;
                        delete div;

                        delete[] maxResultShapeBuffer;
                    }
                    else if (shape::isVector(xShapeBuffer)) {
                        T max = 0;
                        T sum = 0;
                        int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
                        int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);
                        int length = shape::length(xShapeBuffer);
                        if (elementWiseStride >= 1 && resultElementWiseStride >= 1) {
                            if (elementWiseStride == 1 && resultElementWiseStride == 1) {
                                for (int i = 0; i < length; i++) {
                                    max = nd4j::math::nd4j_max<T>(max, dx[i]);
                                }


                                for (int i = 0; i < length; i++) {
                                    result[i] = dx[i] - max;
                                }

                                for (int i = 0; i < length; i++) {
                                    result[i] = nd4j::math::nd4j_exp<T>(result[i]);
                                }


                                for (int i = 0; i < length; i++) {
                                    sum += result[i];
                                }


                                for (int i = 0; i < length; i++) {
                                    result[i] /= sum;
                                }


                            }
                            else {

                                for (int i = 0; i < length; i++) {
                                    max = nd4j::math::nd4j_max<T>(max, dx[i * elementWiseStride]);
                                }
                                for (int i = 0; i < length; i++) {
                                    result[i * resultElementWiseStride] = dx[i * elementWiseStride] - max;
                                }
                                for (int i = 0; i < length; i++) {
                                    result[i * resultElementWiseStride] = nd4j::math::nd4j_exp<T>(
                                            result[i * resultElementWiseStride]);
                                }
                                for (int i = 0; i < length; i++) {
                                    sum += result[i * resultElementWiseStride];
                                }
                                for (int i = 0; i < length; i++) {
                                    result[i * resultElementWiseStride] /= sum;
                                }
                            }

                        }


                    }
                }

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SoftMax() {
                }

#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif

                SoftMax() {
                    this->requiresSpecial = true;
                }


            };


/**
 * softmax(x)
 */
            template<typename T>
            class LogSoftMax : public Transform<T> {
            public:

#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {
		int *shape = shape::shapeOf(xShapeBuffer);
		int *stride = shape::stride(xShapeBuffer);
		//iterate along rows
		int dimension[1] = {0};
		int maxDimension[1] = {1};
		__shared__ functions::reduce::ops::Max<T> *max;
		__shared__ functions::transform::ops::Exp<T> *exp;
		__shared__ functions::broadcast::ops::Subtract<T> *sub;
		__shared__ functions::broadcast::ops::Divide<T> *div;
		__shared__ functions::transform::ops::Log<T> *log;
		__shared__ functions::reduce::ops::Sum<T> *sum;
		__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
		__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
		__shared__ T *maxResult;
		__shared__ int isVector;
		if(threadIdx.x == 0) {
			isVector = shape::isVector(xShapeBuffer);
			max = new functions::reduce::ops::Max<T>();
			exp = new functions::transform::ops::Exp<T>();
			if (isVector) {
				scalarSub = new functions::scalar::ops::Subtract<T>();
				scalarDiv = new functions::scalar::ops::Divide<T>();
			} else {
				sub = new functions::broadcast::ops::Subtract<T>();
				div = new functions::broadcast::ops::Divide<T>();
			}
			log = new functions::transform::ops::Log<T>();
			sum = new functions::reduce::ops::Sum<T>();
			maxResult = (T *) malloc(sizeof(T) * shape[0]);
		}
		__syncthreads();
		//compute the row wise maxes

		if (threadIdx.x < shape[0])
			maxResult[threadIdx.x] = 0.0;
		__syncthreads();

		int maxShape[2] = {shape[0], 1};
		int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);

		max->transformCuda(dx, xShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension, 1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		//subtract max of each row
		if (isVector) {
			scalarSub->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		} else {
			sub->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();

		//after subtracting the row wise maxes take the exp
		exp->transformCuda(result, resultShapeBuffer, extraParams,result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
		__syncthreads();

		//take the sum for the exponential
		sum->transformCuda(result, resultShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension,1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		//divide by the sum
		if (isVector) {
			scalarDiv->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer , allocationPointer, manager);
		} else {
			div->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();


		log->transformCuda(result, resultShapeBuffer, extraParams,result, resultShapeBuffer, allocationPointer, reductionPointer, manager);

		__syncthreads();
		if(threadIdx.x == 0) {
			delete exp;
			delete sum;
			delete max;
			delete log;
			if (isVector) {
				delete scalarDiv;
				delete scalarSub;
			} else {
				delete div;
				delete sub;
			}

			delete[] maxResult;
			delete[] maxResultShapeBuffer;
		}


	}
#endif

                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    if (shape::isMatrix(xShapeBuffer, 2)) {
                        int *shape = shape::shapeOf(xShapeBuffer);
                        //iterate along rows
                        int dimension[1] = {0};
                        int maxDimension[1] = {1};
                        //compute the row wise maxes
                        functions::reduce::ops::Max<T> *max = new functions::reduce::ops::Max<T>();
                        std::vector <T> maxResult(shape[0]);
                        for (int i = 0; i < shape[0]; i++)
                            maxResult[i] = 0.0;
                        int maxShape[2] = {shape[0], 1};
                        int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
                        max->exec(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1);

                        //subtract max of each row
                        functions::broadcast::ops::Subtract<T> *sub = new functions::broadcast::ops::Subtract<T>();
                        sub->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);

                        //after subtracting the row wise maxes take the exp
                        functions::transform::ops::Exp<T> *exp = new functions::transform::ops::Exp<T>();
                        exp->exec(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

                        //take the sum for the exponential
                        functions::reduce::ops::Sum<T> *sum = new functions::reduce::ops::Sum<T>();
                        sum->exec(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1);

                        //divide by the sum
                        functions::broadcast::ops::Divide<T> *div = new functions::broadcast::ops::Divide<T>();
                        div->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);

                        functions::transform::ops::Log<T> *log = new functions::transform::ops::Log<T>();
                        log->exec(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

                        delete exp;
                        delete sub;
                        delete sum;
                        delete max;
                        delete div;
                        delete log;

                        delete[] maxResultShapeBuffer;
                    }
                    else if (shape::isVector(xShapeBuffer, 2)) {
                        T max = 0;
                        T sum = 0;

                        int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
                        int length = shape::length(xShapeBuffer);
                        if (elementWiseStride == 1) {
#pragma omp parallel for shared(max)
                            for (int i = 0; i < length; i++) {
#pragma omp critical
                                {
                                    max = nd4j::math::nd4j_max<T>(max, result[i]);

                                }
                            }
#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] -= max;
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] = nd4j::math::nd4j_exp<T>(result[i]);
                            }

#pragma omp parallel for shared(sum)
                            for (int i = 0; i < length; i++) {
#pragma omp critical
                                {
                                    sum += result[i];

                                }
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] /= sum;
                                result[i] = nd4j::math::nd4j_log<T>(result[i]);
                            }

                        }
                        else {

                            for (int i = 0; i < length; i++) {
                                max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
                            }
#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] -= max;
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
                            }

                            for (int i = 0; i < length; i++) {
                                sum += result[i * elementWiseStride];

                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] /= sum;
                                result[i * elementWiseStride] = nd4j::math::nd4j_log<T>(result[i * elementWiseStride]);
                            }

                        }
                    }
                }

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~LogSoftMax() {
                }

#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif

                LogSoftMax() {
                    this->requiresSpecial = true;
                }


            };


/**
 * softmax(x)
 */
            template<typename T>
            class SoftMaxDerivative : public Transform<T> {
            public:
#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {


		int *shape = shape::shapeOf(xShapeBuffer);
		__shared__ T *maxResult;
		__shared__ int *maxResultShapeBuffer;
		__shared__ int resultEWS;
		__shared__ functions::reduce::ops::Max<T> *max;
		__shared__ functions::transform::ops::Exp<T> *exp;
		__shared__ functions::broadcast::ops::Subtract<T> *sub;
		__shared__ functions::scalar::ops::Subtract<T> *scalarSub;
		__shared__ functions::scalar::ops::Divide<T> *scalarDiv;
		__shared__ functions::broadcast::ops::Divide<T> *div;
		__shared__ functions::reduce::ops::Sum<T> *sum;
		__shared__ int isVector;

		int length = shape::length(xShapeBuffer);

		if(threadIdx.x == 0) {
			isVector = shape::isVector(xShapeBuffer);
			resultEWS = shape::elementWiseStride(resultShapeBuffer);
			max = new functions::reduce::ops::Max<T>();
			sum = new functions::reduce::ops::Sum<T>();
			exp = new functions::transform::ops::Exp<T>();
			if (isVector) {
				scalarSub = new functions::scalar::ops::Subtract<T>();
				scalarDiv = new functions::scalar::ops::Divide<T>();
			} else {
				sub = new functions::broadcast::ops::Subtract<T>();
				div = new functions::broadcast::ops::Divide<T>();
			}
			maxResult = (T *) malloc(sizeof(T) * shape[0]);
		}
		__syncthreads();

		int *stride = shape::stride(xShapeBuffer);
		//iterate along rows
		int dimension[1] = {0};
		int maxDimension[1] = {1};
		//compute the row wise maxes

		int maxShape[2] = {shape[0], 1};

		if (threadIdx.x == 0)
			maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);

		if (threadIdx.x < shape[0])
			maxResult[threadIdx.x] = (T) 0.0;
		__syncthreads();

		max->transformCuda(dx, xShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension, 1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		if (threadIdx.x == 0) delete max;
		__syncthreads();

		//subtract max of each row
		if (isVector) {
			scalarSub->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		} else {
			sub->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();

		//after subtracting the row wise maxes take the exp
		exp->transformCuda(result, resultShapeBuffer, extraParams,result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
		__syncthreads();


		//take the sum for the exponential
		sum->transformCuda(result, resultShapeBuffer, extraParams, maxResult, maxResultShapeBuffer, maxDimension,1,1, allocationPointer, reductionPointer, manager);
		__syncthreads();

		//divide by the sum
		if (isVector) {
			scalarDiv->transformCuda(maxResult[0], result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		} else {
			div->transformCuda(result, resultShapeBuffer, maxResult, maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, manager);
		}
		__syncthreads();

		if (resultEWS >= 1) {
			for (int i = threadIdx.x; i < length; i += blockDim.x) {
				result[i * resultEWS] = result[i * resultEWS] * (1 - result[i * resultEWS]);
			}
		} else {
			printf("Non element wise stride not supported right now\n");
		}

		__syncthreads();
		if(threadIdx.x == 0) {
			delete sum;
			delete exp;
			if (isVector) {
				delete scalarDiv;
				delete scalarSub;
			} else {
				delete div;
				delete sub;
			}

			delete[] maxResult;
			delete[] maxResultShapeBuffer;
		}
	}
#endif


                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    if (shape::isMatrix(xShapeBuffer, 2)) {
                        int *shape = shape::shapeOf(xShapeBuffer);

                        int resultEleStide = shape::elementWiseStride(resultShapeBuffer);

                        //iterate along rows
                        int dimension[1] = {0};
                        int maxDimension[1] = {1};
                        int len = shape::length(xShapeBuffer);
                        //compute the row wise maxes
                        functions::reduce::ops::Max<T> *max = new functions::reduce::ops::Max<T>();
                        std::vector <T> maxResult(shape[0]);
#pragma omp parallel for
                        for (int i = 0; i < shape[0]; i++)
                            maxResult[i] = 0.0;
                        int maxShape[2] = {shape[0], 1};
                        int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
                        max->exec(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1);

                        //subtract max of each row
                        functions::broadcast::ops::Subtract<T> *sub = new functions::broadcast::ops::Subtract<T>();
                        sub->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);

                        //after subtracting the row wise maxes take the exp
                        functions::transform::ops::Exp<T> *exp = new functions::transform::ops::Exp<T>();
                        exp->exec(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

                        //take the sum for the exponential
                        functions::reduce::ops::Sum<T> *sum = new functions::reduce::ops::Sum<T>();
                        sum->exec(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension,
                                  1);

                        //divide by the sum
                        functions::broadcast::ops::Divide<T> *div = new functions::broadcast::ops::Divide<T>();
                        div->exec(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1);

                        if (resultEleStide >= 1) {
                            if (resultEleStide == 1) {
#pragma omp parallel for
                                for (int i = 0; i < len; i++) {
                                    result[i] = result[i] * (1 - result[i]);
                                }

                            }
                            else {
#pragma omp parallel for
                                for (int i = 0; i < len; i++) {
                                    result[i * resultEleStide] = result[i * resultEleStide] * (1 - result[i * resultEleStide]);
                                }

                            }
                        }

                        else {
                            printf("Non element wise stride not supported right now\n");
                        }


                        delete exp;
                        delete sub;
                        delete sum;
                        delete max;
                        delete div;

                        delete[] maxResultShapeBuffer;
                    }
                    else if (shape::isVector(xShapeBuffer, 2)) {
                        T max = 0;
                        T sum = 0;

                        int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
                        int length = shape::length(xShapeBuffer);
                        if (elementWiseStride == 1) {
#pragma omp parallel for shared(max)
                            for (int i = 0; i < length; i++) {
                                max = nd4j::math::nd4j_max<T>(max, result[i]);
                            }
#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] -= max;
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] = nd4j::math::nd4j_exp<T>(result[i]);
                            }

#pragma omp parallel for shared(sum)
                            for (int i = 0; i < length; i++) {
                                sum += result[i];
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i] /= sum;
                            }

                        }
                        else {

#pragma omp parallel for shared(max)
                            for (int i = 0; i < length; i++) {
#pragma omp critical
                                {
                                    max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);

                                }
                            }
#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] -= max;
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
                            }

#pragma omp parallel for shared(sum)
                            for (int i = 0; i < length; i++) {
#pragma omp critical
                                {
                                    sum += result[i * elementWiseStride];
                                }
                            }

#pragma omp parallel for
                            for (int i = 0; i < length; i++) {
                                result[i * elementWiseStride] /= sum;
                            }

                        }
                    }
                }

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif

                virtual ~SoftMaxDerivative() {
                }

#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif

                SoftMaxDerivative() {
                    this->requiresSpecial = true;
                }


            };


/**
 * softmax(x)
 */
            template<typename T>
            class IsMax : public Transform<T> {
            private:

#ifdef __CUDACC__

                inline  __device__ void doAllCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {

		__shared__ functions::indexreduce::ops::IMax<T> *max;
		__shared__ int maxIdx;
		__shared__ int length;
		if(threadIdx.x == 0) {
			max = new functions::indexreduce::ops::IMax<T>();
			length = shape::length(resultShapeBuffer);
		}
		__syncthreads();

		max->transform(
				dx,
				xShapeBuffer,
				extraParams,
				result,
				resultShapeBuffer,
				nullptr,
				1,
				1, allocationPointer, reductionPointer, manager);

		__syncthreads();
		if(threadIdx.x == 0)
			maxIdx = (int) result[0];
		__syncthreads();

		for (int i = threadIdx.x; i < length ; i+= blockDim.x)
			result[i] = 0;
		__syncthreads();

		if (threadIdx.x == 0) {
			result[maxIdx] = 1.0;

			delete max;
		}

	}
#endif

#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)


#endif
                void doAll(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    int length = shape::length(xShapeBuffer);
                    int eleStride = shape::elementWiseStride(xShapeBuffer);
                    int resultEleStride = shape::elementWiseStride(resultShapeBuffer);
                    char xOrder = shape::order(xShapeBuffer);
                    char resultOrder = shape::order(resultShapeBuffer);
                    if (xOrder == resultOrder && xOrder == 'c') {
                        if (eleStride == 1 && resultEleStride == 1) {
                            if (length < 8000) {
                                int maxIdx = 0;
                                T currMax = dx[0];
#pragma omp simd
                                for (int i = 0; i < length; i++) {
                                    if (currMax < dx[i]) {
                                        currMax = dx[i];
                                        maxIdx = i;
                                    }

                                    result[i] = 0.0;

                                }

                                result[maxIdx] = 1.0;

                            }
                            else {
                                int maxIdx = 0;
                                T currMax = dx[0];
#pragma omp parallel for shared(maxIdx,currMax)
                                for (int i = 0; i < length; i++) {
                                    if (currMax < dx[i]) {
                                        currMax = dx[i];
                                        maxIdx = i;
                                    }
                                    result[i] = 0.0;

                                }

                                result[maxIdx] = 1.0;
                            }

                        }
                        else {
                            if (length < 8000) {
                                int maxIdx = 0;
                                T currMax = dx[0];
#pragma omp simd
                                for (int i = 0; i < length; i++) {
                                    result[i * resultEleStride] = 0.0;
                                    if (currMax < dx[i * eleStride]) {
                                        currMax = dx[i * eleStride];
                                        maxIdx = i;
                                    }
                                }

                                result[maxIdx * resultEleStride] = 1.0;

                            }
                            else {
                                int maxIdx = 0;
                                T currMax = dx[0];
#pragma omp parallel for shared(maxIdx,currMax)
                                for (int i = 0; i < length; i++) {
                                    result[i * resultEleStride] = 0.0;
                                    if (currMax < dx[i * eleStride]) {
                                        currMax = dx[i * eleStride];
                                        maxIdx = i;
                                    }
                                }

                                result[maxIdx * resultEleStride] = 1.0;
                            }

                        }
                    }


                    else {
                        int shapeIter[MAX_RANK];
                        int coord[MAX_RANK];
                        int dim;
                        int xStridesIter[MAX_RANK];
                        int resultStridesIter[MAX_RANK];
                        int *xShape = shape::shapeOf(xShapeBuffer);
                        int *xStride = shape::stride(xShapeBuffer);
                        int *resultStride = shape::stride(resultShapeBuffer);
                        int rank = shape::rank(xShapeBuffer);
                        T *originalResult = result;
                        if (PrepareTwoRawArrayIter<T>(rank,
                                                      xShape,
                                                      dx,
                                                      xStride,
                                                      result,
                                                      resultStride,
                                                      &rank,
                                                      shapeIter,
                                                      &dx,
                                                      xStridesIter,
                                                      &result,
                                                      resultStridesIter) >= 0) {
                            T value = dx[0];
                            int idx = 0;
                            int maxIdx = 0;
                            ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                if(dx[0] > value) {
                                    value = dx[0];
                                    maxIdx = idx;
                                }

                                idx++;
                                result[0] = 0.0;

                            }
                            ND4J_RAW_ITER_TWO_NEXT(
                                    dim,
                                    rank,
                                    coord,
                                    shapeIter,
                                    dx,
                                    xStridesIter,
                                    result,
                                    resultStridesIter);

                            //pointer to where max value would be
                            if(shape::order(resultShapeBuffer) == 'c' || (shape::order(resultShapeBuffer) == 'f' &&
                                                                         maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1] >=
                                                                         shape::length(resultShapeBuffer)))
                                originalResult[maxIdx] = 1.0;
                            else
                                originalResult[maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1]] = 1.0;
                        }
                    }


                }
            public:


#ifdef __CUDACC__
                /**
	 *
	 */

	virtual __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory<T> *manager) {
		if(extraParams == nullptr || extraParams[0] == MAX_DIMENSION) {
			this->doAllCuda(dx,xShapeBuffer,result,resultShapeBuffer,extraParams, allocationPointer, reductionPointer, manager);
		} else {
			__shared__ functions::indexreduce::ops::IMax<T> *max;
			__shared__ int maxIdx;
			__shared__ int length;
			if(threadIdx.x == 0) {
				max = new functions::indexreduce::ops::IMax<T>();
				length = shape::length(resultShapeBuffer);
			}

			__syncthreads();

			int dimensionLength = (int) extraParams[0];
			__shared__ int *dimension;
			if(threadIdx.x == 0) {
				dimension = (int *) malloc(sizeof(int) * dimensionLength);
				for(int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int) extraParams[i + 1];
				}
			}

			__syncthreads();

			max->transform(
					dx,
					xShapeBuffer,
					extraParams,
					result,
					resultShapeBuffer,
					dimension,
					dimensionLength,
					1, allocationPointer, reductionPointer, manager);

			__syncthreads();
			if(threadIdx.x == 0) {
				maxIdx = (int) result[0];
			}
			__syncthreads();

			for (int i = threadIdx.x; i < length; i+= blockDim.x)
				result[i] = 0;
			__syncthreads();

			if (threadIdx.x == 0) {
				result[maxIdx] = 1.0;

				delete[] dimension;
				delete max;
			}
		}
	}
#endif
                /**
                 * CPU operation execution
                 * @param dx the input data
                 * @param xStride the stride to iterate over
                 * the x input
                 * @param y the y data
                 * @param yStride the stride to iterate
                 * over the y buffer
                 * @param result the buffer
                 * to store the result in
                 * @param resultStride the stride for the buffer
                 * @param extraParams the extra parameters for the transform
                 * @param n the length of the input
                 */
                virtual void execSpecial(
                        T *dx,
                        int *xShapeBuffer,
                        T *result,
                        int *resultShapeBuffer,
                        T *extraParams) {
                    if (extraParams == nullptr || extraParams[0] == 0 ||
                        (extraParams[0] == 1 && extraParams[1] == MAX_DIMENSION)) {
                        this->doAll(dx, xShapeBuffer, result, resultShapeBuffer, extraParams);
                    }
                    else if(shape::isVector(xShapeBuffer)) {
                        int dimensionLength = (int) extraParams[0];
                        int *dimension = new int[dimensionLength];
                        int length = shape::length(xShapeBuffer);
                        for (int i = 0; i < dimensionLength; i++) {
                            dimension[i] = (int) extraParams[i + 1];
                        }
                        if (shape::shapeOf(xShapeBuffer)[dimension[0]] == 1) {
                            for(int i = 0; i < length; i++) {
                                result[i] = 1.0;
                            }
                        }
                        else {
                            int eleStride = shape::elementWiseStride(xShapeBuffer);
                            if (eleStride == 1) {
                                int maxIdx = 0;
                                T currMax = dx[0];
                                if (length < 8000) {
#pragma omp simd
                                    for (int i = 0; i < length; i++) {
                                        if (currMax < dx[i]) {
                                            currMax = dx[i];
                                            maxIdx = i;
                                        }

                                        dx[i] = 0.0;

                                    }
                                }
                                else {
#pragma omp parallel for shared(maxIdx,currMax)
                                    for (int i = 0; i < length; i++) {
                                        if (currMax < dx[i]) {
                                            currMax = dx[i];
                                            maxIdx = i;
                                        }

                                        result[i] = 0.0;

                                    }
                                }

                                result[maxIdx] = 1.0;

                            }


                            else {
                                int maxIdx = 0;
                                T currMax = dx[0];
                                if (length < 8000) {
#pragma omp simd
                                    for (int i = 0; i < length; i++) {
                                        if (currMax < dx[i * eleStride]) {
                                            currMax = dx[i * eleStride];
                                            maxIdx = i;
                                        }

                                        dx[i] = 0.0;

                                    }
                                }
                                else {
#pragma omp parallel for shared(maxIdx,currMax)
                                    for (int i = 0; i < length; i++) {
                                        if (currMax < dx[i * eleStride]) {
                                            currMax = dx[i * eleStride];
                                            maxIdx = i;
                                        }

                                        result[i] = 0.0;

                                    }
                                }

                                result[maxIdx] = 1.0;

                            }
                        }


                    }
                    else {
                        int dimensionLength = (int) extraParams[0];
                        int *dimension = (int *) malloc(sizeof(int) *dimensionLength);
                        for (int i = 0; i < dimensionLength; i++) {
                            dimension[i] = (int) extraParams[i + 1];
                        }

                        int *shape = shape::shapeOf(xShapeBuffer);
                        int wholeRank = shape::rank(xShapeBuffer);


                        shape::TAD tad(xShapeBuffer,dimension,dimensionLength);
                        tad.createTadOnlyShapeInfo();
                        tad.createOffsets();

                        int tads = tad.numTads;
                        //decompose in to several sub tads after
                        //moving all dimensions (in sorted order)
                        //to the back.
                        //permuted version of the x shape info for setting up the tad problem
                        int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
#pragma omp  parallel  for
                        for (int i = 0; i < tads; i++) {
                            int offset = tad.tadOffsets[i];
                            int shapeIter[MAX_RANK];
                            int coord[MAX_RANK];
                            int dim;
                            int xStridesIter[MAX_RANK];
                            int resultStridesIter[MAX_RANK];
                            int *xShape = shape::shapeOf(tadShapeShapeInfo);
                            int *xStride = shape::stride(tadShapeShapeInfo);
                            int *resultStride = shape::stride(tadShapeShapeInfo);
                            int rank = shape::rank(tadShapeShapeInfo);
                            T *xPointer = dx + offset;
                            T *resultPointer = result + offset;
                            T maxValue = xPointer[0];

                            T *maxCursor = resultPointer;
                            Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                            if (PrepareTwoRawArrayIter<T>(rank,
                                                          xShape,
                                                          xPointer,
                                                          xStride,
                                                          resultPointer,
                                                          resultStride,
                                                          &rank,
                                                          shapeIter,
                                                          &xPointer,
                                                          xStridesIter,
                                                          &resultPointer,
                                                          resultStridesIter) >= 0) {
                                ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                    if (maxValue < xPointer[0]) {
                                        maxCursor = resultPointer;
                                        maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                        maxValue = xPointer[0];
                                    }

                                    resultPointer[0] = 0.0;
                                }
                                ND4J_RAW_ITER_TWO_NEXT(dim,
                                                       rank,
                                                       coord,
                                                       shapeIter,
                                                       xPointer,
                                                       xStridesIter,
                                                       resultPointer,
                                                       resultStridesIter);
                                maxCursor = reinterpret_cast<T *>(maxCursorLong);
                                maxCursor[0] = 1.0;
                            }
                        }

                    }
                }

                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)


#endif
                virtual ~IsMax() {
                }
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                IsMax()
                {
                    this->requiresSpecial = true;
                }


            };

        }


        template<typename T>
        class TransformOpFactory {
        public:
#ifdef __CUDACC__
            __device__ __host__
#endif

            TransformOpFactory() {
            }



            /**
             * Create an op
             * @param op the op to create
             * 0: abs
             * 1: ceiling
             * 2: cosine
             * 3: exp
             * 4: floor
             * 5: log
             * 6: neg
             * 7: pow
             * 8: round
             * 9: setrange
             * 10:sigmoid
             * 11: sign
             * 12: sin
             * 13:softplus
             * 14:sqrt
             * 15:tanh
             * 16:acos
             * 17:asin
             * 18:atan
             * @return the op given the number
             */
#ifdef __CUDACC__
            __inline__ __device__
            Transform<T> * getOp(int op, unsigned char *buffer) {
#else
            Transform<T> * getOp(int op) {
#endif
                /**
                 * We are likely going to need constant symbols for device memory for different operations
                 * or switch to arithmetic based approaches?
                 */
                if (op == 0) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Abs<T>();
#else
                    return new transform::ops::Abs<T>();
#endif
                }
                else if (op == 1) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Ceiling<T>();
#else
                    return new transform::ops::Ceiling<T>();
#endif
                }
                if (op == 2) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Cosine<T>();
#else
                    return new transform::ops::Cosine<T>();
#endif
                }
                else if (op == 3) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Exp<T>();
#else
                    return new transform::ops::Exp<T>();
#endif
                }
                else if (op == 4) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Floor<T>();
#else
                    return new transform::ops::Floor<T>();
#endif
                }
                else if (op == 5) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Log<T>();
#else
                    return new transform::ops::Log<T>();
#endif
                }
                else if (op == 6) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Neg<T>();
#else
                    return new transform::ops::Neg<T>();
#endif
                }
                else if (op == 7) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Pow<T>();
#else
                    return new transform::ops::Pow<T>();
#endif
                }
                else if (op == 8) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Round<T>();
#else
                    return new transform::ops::Round<T>();
#endif
                }
                else if (op == 9) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SetRange<T>();
#else
                    return new transform::ops::SetRange<T>();
#endif
                }
                else if (op == 10) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Sigmoid<T>();
#else
                    return new transform::ops::Sigmoid<T>();
#endif
                }
                else if (op == 11) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Sign<T>();
#else
                    return new transform::ops::Sign<T>();
#endif
                }
                else if (op == 12) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Sin<T>();
#else
                    return new transform::ops::Sin<T>();
#endif
                }
                else if (op == 13) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SoftPlus<T>();
#else
                    return new transform::ops::SoftPlus<T>();
#endif
                }
                else if (op == 14) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Sqrt<T>();
#else
                    return new transform::ops::Sqrt<T>();
#endif
                }
                else if (op == 15) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Tanh<T>();
#else
                    return new transform::ops::Tanh<T>();
#endif
                }
                else if (op == 16) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::ACos<T>();
#else
                    return new transform::ops::ACos<T>();
#endif
                }
                else if (op == 17) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::ASin<T>();
#else
                    return new transform::ops::ASin<T>();
#endif
                }
                else if (op == 18) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::ATan<T>();
#else
                    return new transform::ops::ATan<T>();
#endif
                }
                else if (op == 19) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::HardTanh<T>();
#else
                    return new transform::ops::HardTanh<T>();
#endif
                }
                else if (op == 20) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SoftSign<T>();
#else
                    return new transform::ops::SoftSign<T>();
#endif
                }
                else if (op == 21) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::ELU<T>();
#else
                    return new transform::ops::ELU<T>();
#endif
                }
                else if (op == 22) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::ELUDerivative<T>();
#else
                    return new transform::ops::ELUDerivative<T>();
#endif
                }
                else if (op == 23) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::TanhDerivative<T>();
#else
                    return new transform::ops::TanhDerivative<T>();
#endif
                }
                else if (op == 24) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::TimesOneMinus<T>();
#else
                    return new transform::ops::TimesOneMinus<T>();
#endif
                }
                else if(op == 25) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::HardTanhDerivative<T>();
#else
                    return new transform::ops::HardTanhDerivative<T>();
#endif
                }
                else if(op == 26) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Ones<T>();
#else
                    return new transform::ops::Ones<T>();
#endif
                }
                else if(op == 27) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Identity<T>();
#else
                    return new transform::ops::Identity<T>();
#endif
                }
                else if(op == 28) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Stabilize<T>();
#else
                    return new transform::ops::Stabilize<T>();
#endif
                }
                else if(op == 29) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SigmoidDerivative<T>();
#else
                    return new transform::ops::SigmoidDerivative<T>();
#endif
                }
                else if(op == 30) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SoftSignDerivative<T>();
#else
                    return new transform::ops::SoftSignDerivative<T>();
#endif
                }
                else if(op == 31) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::LeakyRELU<T>();
#else
                    return new transform::ops::LeakyRELU<T>();
#endif
                }
                else if(op == 32) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::LeakyRELUDerivative<T>();
#else
                    return new transform::ops::LeakyRELUDerivative<T>();
#endif
                }
                else if(op == 33) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::RELU<T>();
#else
                    return new transform::ops::RELU<T>();
#endif
                }
                else if(op == 34) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Step<T>();
#else
                    return new transform::ops::Step<T>();
#endif
                }
                else if(op == 35) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::OneMinus<T>();
#else
                    return new transform::ops::OneMinus<T>();
#endif
                }
                else if(op == 36) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Col2Im<T>();
#else
                    return new transform::ops::Col2Im<T>();
#endif
                }
                else if(op == 37) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::Im2col<T>();
#else
                    return new transform::ops::Im2col<T>();
#endif
                }
                else if(op == 38) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SoftMax<T>();
#else
                    return new transform::ops::SoftMax<T>();
#endif
                }
                else if(op == 39) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SoftMaxDerivative<T>();
#else
                    return new transform::ops::SoftMaxDerivative<T>();
#endif
                }
                else if(op == 40) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::LogSoftMax<T>();
#else
                    return new transform::ops::LogSoftMax<T>();
#endif
                }
                else if(op == 41) {
#ifdef __CUDACC__
                    return new(buffer) transform::ops::IsMax<T>();
#else
                    return new transform::ops::IsMax<T>();
#endif
                } else if(op == 42) {
                    // temporary special op for blockwise SoftMax Derivative
#ifdef __CUDACC__
                    return new(buffer) transform::ops::SpecialDerivative<T>();
#else
                    return new transform::ops::SpecialDerivative<T>();
#endif
                }

                return nullptr;
            }

        };
    }
}




#ifdef __CUDACC__
/*
 * 	T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes
 */



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		Nd4jIndex n,
		T *dy,
		int incy,
		T *params,
		T *result,
		int resultStride, int *allocationPointer, T *reductionPointer) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;

	__shared__ UnifiedSharedMemory<T> *manager;

	if(threadIdx.x == 0) {
	    extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory<T>();
	    manager->init(sizeof(UnifiedSharedMemory<T>), sizeof(functions::transform::TransformOpFactory<T>), sizeof(functions::transform::ops::SoftMaxDerivative<T>), sizeof(shape::TAD));

        manager->setXSpace(0);
	    manager->setYSpace(0);
	    manager->setZSpace(0);
	    manager->setTADSpace(0);

		doubleTransformFactory = new(manager->getFactorySpace()) functions::transform::TransformOpFactory<T>();
        op = doubleTransformFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();

	op->transformCuda(n,dy,incy,params,result,resultStride,allocationPointer, reductionPointer, manager);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
__global__ void transformDouble(
		int opNum,
		Nd4jIndex n,
		double *dy,
		int incy,
		double *params,
		double *result,int resultStride, int *allocationPointer, double *reductionPointer) {

	transformGeneric<double>(
			opNum,
			n,
			dy,
			incy,
			params,
			result,
			resultStride, allocationPointer, reductionPointer);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
__global__ void transformFloat(
		int opNum,
		Nd4jIndex n,
		float *dy,
		int incy,
		float *params,
		float *result,int resultStride, int *allocationPointer, float *reductionPointer) {

	transformGeneric<float>(
			opNum,
			n,
			dy,
			incy,
			params,
			result,resultStride, allocationPointer, reductionPointer);

}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;

	__shared__ UnifiedSharedMemory<T> *manager;

	__shared__ int *ptrSharedXShapeInfo;
    __shared__ int *ptrSharedZShapeInfo;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory<T>();

        manager->setXSpace(xRank);
	    manager->setYSpace(0);
	    manager->setZSpace(zRank);
	    manager->setTADSpace(0);

	    manager->init(sizeof(UnifiedSharedMemory<T>), sizeof(functions::transform::TransformOpFactory<T>), sizeof(functions::transform::ops::SoftMaxDerivative<T>), sizeof(shape::TAD));
    }
    __syncthreads();


	if (xShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(xShapeInfo, manager->getXShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedXShapeInfo = manager->getXShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedXShapeInfo = nullptr;

    if (resultShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(resultShapeInfo, manager->getZShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedZShapeInfo = manager->getZShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedZShapeInfo = nullptr;

	if(threadIdx.x == 0) {
		doubleTransformFactory = new(manager->getFactorySpace()) functions::transform::TransformOpFactory<T>();
		op = doubleTransformFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();


	op->transformCuda(
	    dy,
	    ptrSharedXShapeInfo,
	    params,
	    result,
	    ptrSharedZShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    manager);
}



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformDouble(
		int opNum,
		double *dy,
		int *shapeInfo, int xRank,
		double *params,
		double *result,int *resultShapeInfo, int zRank, int *allocationPointer, double *reductionPointer) {

	transformGeneric<double>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,resultShapeInfo, zRank, allocationPointer, reductionPointer);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformFloat(
		int opNum,
		float *dy,
		int *shapeInfo, int xRank,
		float *params,
		float *result,int *resultShapeInfo, int zRank, int *allocationPointer, float *reductionPointer) {

	transformGeneric<float>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,
			resultShapeInfo, zRank, allocationPointer, reductionPointer);

}



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
template <typename T>
__device__ void transformGenericIndexes(
		int opNum,
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *indexes, int *allocationPointer, T *reductionPointer) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;

	__shared__ UnifiedSharedMemory<T> *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory<T>();
	    manager->init(sizeof(UnifiedSharedMemory<T>), sizeof(functions::transform::TransformOpFactory<T>), sizeof(functions::transform::ops::SoftMaxDerivative<T>), sizeof(shape::TAD));

	    manager->setXSpace(xRank);
	    manager->setYSpace(0);
	    manager->setZSpace(0);
	    manager->setTADSpace(0);
    }
    __syncthreads();

	__shared__ int *ptrSharedXShapeInfo;

	if (xShapeInfo != nullptr) {
    	shape::sweepShapeInfoBuffer(xShapeInfo, manager->getXShapeBuffer());
    	if (threadIdx.x == 0) ptrSharedXShapeInfo = manager->getXShapeBuffer();
    } else if (threadIdx.x == 0) ptrSharedXShapeInfo = nullptr;

	if(threadIdx.x == 0) {
		doubleTransformFactory = new(manager->getFactorySpace()) functions::transform::TransformOpFactory<T>();
		op = doubleTransformFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();


	op->transformCuda(dy,ptrSharedXShapeInfo,params,result,indexes,allocationPointer, reductionPointer, manager);
}



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformDoubleIndexes(
		int opNum,
		double *dy,
		int *shapeInfo, int xRank,
		double *params,
		double *result,int *indexes, int *allocationPointer, double *reductionPointer) {

	transformGenericIndexes<double>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,indexes, allocationPointer, reductionPointer);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformFloatIndexes(
		int opNum,
		float *dy,
		int *shapeInfo, int xRank,
		float *params,
		float *result,int *indexes, int *allocationPointer, float *reductionPointer) {

	transformGenericIndexes<float>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,indexes, allocationPointer, reductionPointer);

}

/**
* This is utility kernel, that updates given special buffer with proper values in device memory
*/
extern "C" __global__ void prepareShapeBuffer(int *dimension, int *maxDimension, int *specialPointer, int rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    dimension[0] = 0;
    maxDimension[0] = 1;

    specialPointer[0] = 2;
    specialPointer[1] = rows;
    specialPointer[2] = 1;
    specialPointer[3] = 1;
    specialPointer[4] = 1;
    specialPointer[5] = 0;
    specialPointer[6] = 1;
    specialPointer[7] = 99;
}
template <typename T>
__device__ void fillIsMaxGeneric(T *dx, long length, long idx) {

   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (long i = tid; i < length; i+= blockDim.x * gridDim.x) {
        dx[i] = (i == idx? 1.0 : 0.0);
   }
}

extern "C" __global__ void fillIsMaxFloat(float *dx, long length, long idx) {
    fillIsMaxGeneric<float>(dx, length, idx);
}

extern "C" __global__ void fillIsMaxDouble(double *dx, long length, long idx) {
    fillIsMaxGeneric<double>(dx, length, idx);
}

#endif

#endif /* TRANSFORM_H_ */
