/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#define EXTRA_PARAMS_LENGTH 10

#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/sharedmem.h>
#ifndef __CUDACC__
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>
#include <helpers/shape.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif


#define REDUCE3_OPS \
        (0, simdOps::ManhattanDistance), \
        (1, simdOps::EuclideanDistance), \
        (2, simdOps::CosineSimilarity), \
        (3, simdOps::Dot), \
        (4, simdOps::EqualsWithEps)


namespace functions {
    namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
        template<typename T>
        class Reduce3 {

        public:
#ifdef __CUDACC__
            virtual __device__

			inline T opAtomic(T d1, T d2, T *extraParamsRef) = 0;
#endif

#ifdef __CUDACC__
            /**
     * Aggregate shared memory
     * @param sPartialsRef
     * @param tid
     * @param extraParams
     */
template<typename OpType>
			static __inline__ __device__ void aggregatePartials(T **sPartialsRef, int tid, int numItems, T *extraParamsRef) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				T *sPartials = *sPartialsRef;
				int floorPow2 = numItems;

				if (floorPow2 & (floorPow2 - 1)) {
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}
					if (tid >= floorPow2) {
						sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParamsRef);
					}
					__syncthreads();
				}

				for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads) {
						sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParamsRef);
					}
					__syncthreads();
				}
			}
			/**

                     Perform a reduction
                     @param n the number of elements
                     @param xOffset the starting offset
                     @param dx the data to perform the reduction on
                     @param incx the increment on which to perform the reduction
                     @param extraParams extra parameters used for calculations
                     @param result where to store the result of the reduction
             */
			virtual __inline__ __device__ void transformNoElementWiseStride(
					T *dx,
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo,
					int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
				Nd4jIndex n = shape::length(xShapeInfo);
				int rank = shape::rank(xShapeInfo);

				T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
				T startingVal = this->startingValue(dx);

				// FIXME: this ugly fast fix.
				__shared__ T extraZ[2];
				if (threadIdx.x == 0) {
					extraZ[0] = (T) 0.0;
					extraZ[1] = (T) 0.0;
				}
				sPartials[threadIdx.x] = startingVal;
				__syncthreads();

                int idx[MAX_RANK];

				for(Nd4jIndex i = blockIdx.x * gridDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(rank,shape::shapeOf(xShapeInfo),i, idx);
					Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),idx,rank);
					Nd4jIndex yOffset = shape::getOffset(0,shape::shapeOf(yShapeInfo),shape::stride(yShapeInfo),idx,rank);
					sPartials[threadIdx.x] = update(sPartials[threadIdx.x], this->opAtomic(dx[offset], dy[yOffset], extraZ), extraZ);
				}

				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), extraZ);
				/**
                 * Look at something that uses the extra params
                 * and aggregates the extra values propelry.
                 *This will be used in summary stats too.
                 */
				// write result for this block to global mem
				if (threadIdx.x == 0) {
					if (postProcessOrNot) {
						result[blockIdx.x] = postProcess(sPartials[0], n, extraZ);
					}
					else {
						result[blockIdx.x] = sPartials[0];
					}


				}
			}


			/**
             *
             */
template<typename OpType>
			static inline __device__ void execScalarCuda(
					T *dx,
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo, int *allocationPointer, T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {


//		SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); // val.getPointer();

				// FIXME: this ugly fast fix.
				__shared__ T extraZ[3];
				if (threadIdx.x == 0) {
					extraZ[0] = (T) 0.0f;
					extraZ[1] = (T) 0.0f;

					if (extraParams != NULL) {
                        extraZ[2] = extraParams[0];
                    } else extraZ[2] = (T) 0.0f;
				}

				__syncthreads();

				T startingVal = OpType::startingValue(dx);
				Nd4jIndex length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
				int tid = blockIdx.x * blockDim.x + threadIdx.x;
				char xOrder = shape::order(xShapeInfo);
				char yOrder = shape::order(yShapeInfo);

				if(xOrder == yOrder && (xElementWiseStride > 0 && yElementWiseStride > 0)) {
					if (xElementWiseStride == 1 && yElementWiseStride == 1) {
						for(Nd4jIndex i = tid; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i], dy[i], extraZ), extraZ);
						}
					}
					else {
						for(Nd4jIndex i = tid; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i * xElementWiseStride], dy[i * yElementWiseStride], extraZ), extraZ);
						}
					}

					sPartials[threadIdx.x] = startingVal;
				} else {
				    __shared__ int *xShape;
				    __shared__ int *yShape;
				    __shared__ int *xStride;
				    __shared__ int *yStride;
				    __shared__ int xElementWiseStride;
				    __shared__ int yElementWiseStride;
				    __shared__ int rank;
				    __shared__ Nd4jIndex length;
				    if (threadIdx.x == 0) {

					    xShape = shape::shapeOf(xShapeInfo);
					    yShape = shape::shapeOf(yShapeInfo);
					    xStride = shape::stride(xShapeInfo);
					    yStride = shape::stride(yShapeInfo);
					    xElementWiseStride = shape::elementWiseStride(xShapeInfo);
					    yElementWiseStride = shape::elementWiseStride(yShapeInfo);
					    rank = shape::rank(xShapeInfo);
					    length = shape::length(xShapeInfo);
					}
					__syncthreads();
					T startingVal = OpType::startingValue(dx);

					T *sPartials = (T *) manager->getSharedReductionBuffer();

					int xCoords[MAX_RANK];
					int yCoords[MAX_RANK];

					sPartials[threadIdx.x] = startingVal;

					for(unsigned int i = tid ;i < length; i += gridDim.x * blockDim.x) {
						shape::ind2subC(rank,xShape,i,xCoords);
						shape::ind2subC(rank,yShape,i,yCoords);

						Nd4jIndex offset = shape::getOffset(0, xShape, xStride, xCoords,rank);
						Nd4jIndex yOffset = shape::getOffset(0,yShape, yStride, yCoords,rank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[offset], dy[yOffset], extraZ), extraZ);
					}
				}

				__syncthreads();

				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, length), extraZ);

				__syncthreads();
				if (gridDim.x > 1) {
                    unsigned int *tc = (unsigned int *)reductionBuffer;
					__shared__ bool amLast;
					int rank = shape::rank(xShapeInfo);
					tid = threadIdx.x;
					T *extraBuffer = (T *) allocationPointer;
					if (threadIdx.x == 0) {
						reductionBuffer[blockIdx.x] = sPartials[0];
						extraBuffer[blockIdx.x] = extraZ[0];
						extraBuffer[gridDim.x + blockIdx.x] = extraZ[1];
					}
					__threadfence();
					__syncthreads();

					if (threadIdx.x == 0) {
						unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
						amLast = (ticket == gridDim.x - 1);
					}

                    sPartials[tid] = startingVal;
					__syncthreads();

					if (amLast) {
						tc[16384] = 0;

						sPartials[threadIdx.x] = OpType::startingValue(dx);

                        // TODO: later probably replace this. Right now we need extraZ sync for CosineSimilarity ONLY
						if (tid == 0 && extraZ[0] != (T) 0.0 && extraZ[1] != (T) 0.0) {
						    extraZ[0] = 0.0;
						    extraZ[1] = 0.0;
                            for (int i = 0; i < gridDim.x; i++) {
                                extraZ[0] += extraBuffer[i];
                                extraZ[1] += extraBuffer[gridDim.x + i];
                            }
						}

						for (Nd4jIndex i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
							sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraZ);
						}
						__syncthreads();

						aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraZ);

						__syncthreads();
						if (threadIdx.x == 0) {
							result[0] = OpType::postProcess(sPartials[0], length, extraZ);
						}
					}
				} else {
					if (tid == 0) {
					    unsigned int *tc = (unsigned *)reductionBuffer;
					    tc[16384] = 0;

						result[0] = OpType::postProcess(sPartials[0], length, extraZ);
					}
				}


			}
			/**
                     Perform a reduction
                     @param n the number of elements
                     @param xOffset the starting offset
                     @param dx the data to perform the reduction on
                     @param incx the increment on which to perform the reduction
                     @param extraParams extra parameters used for calculations
                     @param result where to store the result of the reduction
             */
template<typename OpType>
			__device__
			static inline void transform(
					T *dx,
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo,
					int *dimension,
					int dimensionLength,
					int postProcessOrNot,
					int *allocationPointer,
					UnifiedSharedMemory *manager,
					int *tadOnlyShapeInfo,
					int *tadOffsets,
					int *yTadOnlyShapeInfo,
					int *yTadOffsets) {
				/**
                 * Gpu information for the problem
                 */
				int tid = threadIdx.x + blockIdx.x * blockDim.x;

				__shared__ int resultScalar;

				__shared__ int xElementWiseStride;
				__shared__ int yElementWiseStride;
				//shared memory space for storing intermediate results
				//SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
				T init = OpType::startingValue(dx);
				sPartials[threadIdx.x] = init;

				// FIXME: this ugly fast fix.
				__shared__ T extraZ[2];
				if (threadIdx.x == 0) {
					extraZ[0] = (T) 0.0;
					extraZ[1] = (T) 0.0;
				}
				__syncthreads();
				//length for the tad

				__shared__ Nd4jIndex resultLength;
				__shared__ int tadLength;
				__shared__ int tadElementWiseStride;
				__shared__ int yTadElementWiseStride;


				T reduction = OpType::startingValue(dx);
				if (threadIdx.x == 0) {
					if (resultShapeInfo != nullptr)
						resultLength = shape::length(resultShapeInfo);
					else resultLength = 1;

					if (dimensionLength == 1) {
						if (dimension == nullptr || dimension[0] == MAX_DIMENSION)
							resultScalar = 1;
						else
							resultScalar = 0;
					}
					else
						resultScalar = 0;

					if (resultLength == 1)
						resultScalar = 1;

					int *xStride = shape::stride(xShapeInfo);
					char xOrder = shape::order(xShapeInfo);

					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadElementWiseStride = shape::elementWiseStride(tadOnlyShapeInfo);
					yTadElementWiseStride = shape::elementWiseStride(yTadOnlyShapeInfo);
				}
				__syncthreads();


				if (!resultScalar) {
					if(tadElementWiseStride >= 1 && yTadElementWiseStride) {
    					for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
							int xOffsetForTad = tadOffsets[i];
							int yOffsetForTad = yTadOffsets[i];

                            if (threadIdx.x < tadLength)
							    sPartials[threadIdx.x] =  OpType::op(dx[xOffsetForTad + tadElementWiseStride * threadIdx.x],dy[yOffsetForTad + yTadElementWiseStride * threadIdx.x], extraZ);

							for(int j = threadIdx.x + blockDim.x; j < tadLength; j += blockDim.x) {
								sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffsetForTad + tadElementWiseStride * j],dy[yOffsetForTad + yTadElementWiseStride * j], extraZ), extraZ);
							}
							__syncthreads();

                            T **sPartialsRef = (T **) &sPartials;
				            aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                            __syncthreads();
                            if (threadIdx.x == 0)
							    result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);
						}
					}
					else {
/*
						// DO NOT REMOVE THIS COMMENTED BLOCK PLEASE

						for (int r = blockIdx.x; r < tad->numTads; r += gridDim.x) {
                            if (threadIdx.x == 0)
                                tad->createOffsetForBlock(r);
                            __syncthreads();

                            int tadOffsetForBlock = tad->tadOffsetForBlock;
                            T *xVal = dx + tadOffsetForBlock;


                            sPartials[threadIdx.x] = this->startingValue(xVal);
                            for(int i = threadIdx.x; i < tad->tadLength; i+= blockDim.x) {
                    			int xOffsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength, nullptr);
								int yOffsetForTad = shape::tadOffset(i, yShapeInfo, dimension, dimensionLength, nullptr);

                                sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],dx[tadOffsetForBlock + i *  tad->tadElementWiseStride], extraParams);
                            }
                            __syncthreads();

                            // aggregate. do NOT reduce for elements > tadLength
                            T **sPartialsRef = (T **) &sPartials;
                            aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength), extraParams);


                            __syncthreads();
                            if (threadIdx.x == 0)
                                result[r] = this->postProcess(sPartials[threadIdx.x], tad->tadLength, extraParams);
                        }

*/

                        int xCoord[MAX_RANK];
                        int yCoord[MAX_RANK];

                        int *yShape = shape::shapeOf(yTadOnlyShapeInfo);
						int *yStride = shape::stride(yTadOnlyShapeInfo);
						int *xShape = shape::shapeOf(tadOnlyShapeInfo);
						int *xStride = shape::stride(tadOnlyShapeInfo);
						int yRank = shape::rank(yTadOnlyShapeInfo);
						int xRank = shape::rank(tadOnlyShapeInfo);


						for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
							int xOffsetForTad = tadOffsets[i];
							int yOffsetForTad = yTadOffsets[i];

							for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                                shape::ind2subC(xRank,xShape, j, xCoord);
                                shape::ind2subC(yRank,yShape, j, yCoord);

                                Nd4jIndex xOffset = shape::getOffset(xOffsetForTad, xShape, xStride, xCoord, xRank);
                                Nd4jIndex yOffset = shape::getOffset(yOffsetForTad, yShape, yStride, yCoord, yRank);

								sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ), extraZ);
							}
							__syncthreads();

                            T **sPartialsRef = (T **) &sPartials;
				            aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                            __syncthreads();
                            if (threadIdx.x == 0)
							    result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);
						}

					}
				}
			}





#endif

#ifdef __CUDACC__
            __device__
			static inline void exec(
				const int opNum,
				T *dx,
				int *xShapeInfo,
				T *dy,
				int *yShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				int postProcessOrNot,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo,
				int *tadOffsets,
				int *yTadOnlyShapeInfo,
				int *yTadOffsets) {
                            DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
			}

			__device__
			static inline void execScalarCuda(
				const int opNum,
				T *dx,
				int *xShapeInfo,
				T *dy,
				int *yShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int * allocationPointer,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
                            DISPATCH_BY_OPNUM(execScalarCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, allocationPointer, reductionBuffer, manager, tadOnlyShapeInfo), REDUCE3_OPS);
			}
#endif


#ifdef __CUDACC__
            __host__
#endif

            static T execScalar(
                    const int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *extraParamsVals,
                    T *y,
                    int *yShapeInfo) {
                RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x,
                                                               xShapeInfo,
                                                               extraParamsVals,
                                                               y,
                                                               yShapeInfo), REDUCE3_OPS);
            }

            static void exec( const int opNum,
                              T *x, int *xShapeInfo,
                              T *extraParamsVals,
                              T *y,
                              int *yShapeInfo,
                              T *result,
                              int *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParamsVals,
                                               y, yShapeInfo,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength), REDUCE3_OPS);
            }


            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static T execScalar(
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *y,
                    int *yShapeInfo) {
                T startingVal = OpType::startingValue(x);
                Nd4jIndex length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int yElementWiseStride = shape::elementWiseStride(yShapeInfo);

                T extraParamsVals[3] = {(T) 0.0, (T) 0.0, (T) 0.0};
                // it's possible case for EqualsWithEps op
                if (extraParams != nullptr) {
                    extraParamsVals[2] = extraParams[0];
                }


                char xOrder = shape::order(xShapeInfo);
                char yOrder = shape::order(yShapeInfo);
                if(xOrder == yOrder && (xElementWiseStride  >= 1 && yElementWiseStride >= 1)) {
                    if (xElementWiseStride == 1 && yElementWiseStride == 1) {

// TODO:: proper reduction required here
                        for(int i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal,
                                                         OpType::op(x[i],y[i],
                                                                    extraParamsVals),
                                                         extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);

                    }

                    else {
// TODO:: proper reduction required here
                        for(Nd4jIndex i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal, OpType::op(x[i * xElementWiseStride],y[i * yElementWiseStride], extraParamsVals), extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);
                    }

                }


                else {
                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    int *yStride = shape::stride(yShapeInfo);
                    T startingVal = OpType::startingValue(x);
                    Nd4jIndex n = shape::length(xShapeInfo);
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int yStridesIter[MAX_RANK];
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 x,
                                                 xStride,
                                                 y,
                                                 yStride,
                                                 &rank,
                                                 shapeIter,
                                                 &x,
                                                 xStridesIter,
                                                 &y,
                                                 yStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                startingVal = OpType::update(startingVal, OpType::op(x[0],y[0], extraParamsVals), extraParamsVals);
                            } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     x,
                                                     xStridesIter,
                                                     y,
                                                     yStridesIter);

                        return OpType::postProcess(startingVal, n, extraParamsVals);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                }

                return startingVal;


            }

            template<typename OpType>
            static void exec(
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *y,
                    int *yShapeInfo,
                    T *result,
                    int *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength) {

                T extraParamsVals[2] = {(T) 0.0, (T) 0.0};


                if(shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar<OpType>(
                            x,
                            xShapeInfo,
                            extraParamsVals,
                            y,
                            yShapeInfo);
                    return;
                }



                char xOrder = shape::order(xShapeInfo);
                char yOrder = shape::order(yShapeInfo);
                if(xOrder != yOrder) {
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int yStridesIter[MAX_RANK];

                    int *xShape = shape::shapeOf(xShapeInfo);

                    int *xStride = shape::stride(xShapeInfo);
                    int *yStride = shape::stride(yShapeInfo);

                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 x,
                                                 xStride,
                                                 y,
                                                 yStride,
                                                 &rank,
                                                 shapeIter,
                                                 &x,
                                                 xStridesIter,
                                                 &y,
                                                 yStridesIter) >= 0) {

                        Nd4jIndex resultLength = shape::length(resultShapeInfoBuffer);
                        Nd4jIndex tadLength = shape::tadLength(xShapeInfo,dimension,dimensionLength);
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                Nd4jIndex xOffset = shape::getOffset(0,xShape,xStride,coord,rank);
                                int reductionIndex = xOffset / resultLength;
                                result[reductionIndex] = OpType::update(result[reductionIndex], OpType::op(x[0],y[0], extraParamsVals), extraParamsVals);
                            } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     x,
                                                     xStridesIter,
                                                     y,
                                                     yStridesIter);


#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
                        for(Nd4jIndex i = 0; i < resultLength ;i++) {
                            result[i] = OpType::postProcess(result[i],tadLength, extraParamsVals);
                        }
                    }

                    else {
                        printf("Unable to prepare array\n");
                    }
                }
                else {
                    T startingVal = OpType::startingValue(x);

                    Nd4jIndex resultLength = shape::length(resultShapeInfoBuffer);
                    shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                    xTad.createTadOnlyShapeInfo();
                    xTad.createOffsets();


                    shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                    yTad.createTadOnlyShapeInfo();
                    yTad.createOffsets();

                    /**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */
                    int tadElementWiseStride = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
                    int yElementWiseStride = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
                    int tadLength = shape::length(xTad.tadOnlyShapeInfo);
                    if (tadElementWiseStride >= 1 && yElementWiseStride >= 1) {
#pragma omp parallel for proc_bind(AFFINITY) default(shared)
                        for (Nd4jIndex i = 0; i < resultLength; i++) {
                            T *localExtraParams = nullptr;
                            if (OpType::extraParamsLen > 0)
                                localExtraParams = new T[OpType::extraParamsLen];
                            for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                                localExtraParams[extraParamsIdx] = startingVal;
                            }

                            Nd4jIndex offset = xTad.tadOffsets[i];
                            Nd4jIndex yOffset = yTad.tadOffsets[i];
                            result[i] = OpType::op(x[offset], y[yOffset], localExtraParams);
                            for (int j = 1; j < tadLength; j++) {
                                result[i] = OpType::update(result[i], OpType::op(x[offset + tadElementWiseStride * j],
                                                                                 y[yOffset + yElementWiseStride * j],
                                                                                 localExtraParams), localExtraParams);
                            }

                            result[i] = OpType::postProcess(result[i], tadLength, localExtraParams);

                            if (localExtraParams != nullptr)
                                delete[] localExtraParams;
                        }
                    } else {
                        int shapeIter[MAX_RANK];
                        int coord[MAX_RANK];
                        int dim;
                        int xStridesIter[MAX_RANK];
                        int yStridesIter[MAX_RANK];
                        shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                        xTad.createTadOnlyShapeInfo();
                        xTad.createOffsets();


                        shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                        yTad.createTadOnlyShapeInfo();
                        yTad.createOffsets();
                        int tads = shape::tensorsAlongDimension(xShapeInfo,dimension,dimensionLength);
                        int idx[MAX_RANK];
                        int tadsPerThread = resultLength / TAD_THRESHOLD;
                        int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());


#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                        for (int i = 0; i < resultLength; i++) {
                            Nd4jIndex xOffset = xTad.tadOffsets[i];
                            Nd4jIndex yOffset = yTad.tadOffsets[i];
                            int coord[MAX_RANK];

                            T start = OpType::startingValue(x + xOffset);

                            for (int j = 0; j < shape::length(xTad.tadOnlyShapeInfo); j++) {
                                shape::ind2subC(shape::rank(xTad.tadOnlyShapeInfo), shape::shapeOf(xTad.tadOnlyShapeInfo), j, coord);
                                int xOffset2 = shape::getOffset(xOffset,shape::shapeOf(xTad.tadOnlyShapeInfo),shape::stride(xTad.tadOnlyShapeInfo),coord,shape::rank(xTad.tadOnlyShapeInfo));
                                int yOffset2 = shape::getOffset(yOffset,shape::shapeOf(yTad.tadOnlyShapeInfo),shape::stride(yTad.tadOnlyShapeInfo),coord,shape::rank(yTad.tadOnlyShapeInfo));
                                start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParams), extraParams);
                            }

                            result[i] = OpType::postProcess(start, shape::length(xTad.tadOnlyShapeInfo), extraParams);
                        }
                    }


                }
            }


        };
    }
}

#ifdef __CUDACC__
/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post
 */
template <typename T>
__device__ void reduce3Generic(
		const int opNum,
		T *dx,
		int *xShapeInfo,
		T *dy,
		int *yShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::exec(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationPointer,
			manager,
			tadOnlyShapeInfo,
			tadOffsets,
			yTadOnlyShapeInfo,
			yTadOffsets);
}

template <typename T>
__device__ void reduce3ScalarGeneric(
		int opNum,
		T *dx,
		int *xShapeInfo,
		T *dy,
		int *yShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo, int *allocationPointer,
		T *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::execScalarCuda(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			allocationPointer,
			reductionBuffer,
			manager,
			tadOnlyShapeInfo);
}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C"
__global__ void reduce3Double(
		int opNum,
		double *dx,
		int *xShapeInfo,
		double *dy,
		int *yShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3Generic<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C"
__global__ void reduce3Float(
		int opNum,
		float *dx,
		int *xShapeInfo,
		float *dy,
		int *yShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3Generic<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3Half(
		int opNum,
		float16 *dx,
		int *xShapeInfo,
		float16 *dy,
		int *yShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3Generic<float16>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3ScalarFloat(
		int opNum,
		float *dx,
		int *xShapeInfo,
		float *dy,
		int *yShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, float *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3ScalarGeneric<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C" __global__ void reduce3ScalarHalf(
		int opNum,
		float16 *dx,
		int *xShapeInfo,
		float16 *dy,
		int *yShapeInfo,
		float16 *extraParams,
		float16 *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, float16 *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3ScalarGeneric<float16>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3ScalarDouble(
		int opNum,
		double *dx,
		int *xShapeInfo,
		double *dy,
		int *yShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, double *reductionBuffer, int *tadOnlyShapeInfo, int *tadOffsets, int *yTadOnlyShapeInfo, int *yTadOffsets) {
	reduce3ScalarGeneric<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

#endif



#endif /* REDUCE3_H_ */
