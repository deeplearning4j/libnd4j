//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/reduce.h>
#include <loops/legacy_ops.h>


template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }
    __syncthreads();


    functions::reduce::ReduceFunction<T>::template transformCudaXD<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo,
            tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric1D(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer,
        int *tadOnlyShapeInfo,
        Nd4jIndex *tadOffsets) {

    functions::reduce::ReduceFunction<T>::template transformCuda1D<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer, nullptr, tadOnlyShapeInfo, tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric3D(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer,
        int *tadOnlyShapeInfo,
        Nd4jIndex *tadOffsets) {

    functions::reduce::ReduceFunction<T>::template transformCuda3D<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer,
            nullptr,
            tadOnlyShapeInfo,
            tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceScalarGeneric(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer, int *tadOnlyShapeInfo) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), 0);
    }
    __syncthreads();

    functions::reduce::ReduceFunction<T>::template execScalarCuda<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo);
};

// reduceScalar
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float, INPUT(float *x, int *xShapeInfo, float *extraParams, float *z, int *zShapeInfo, int *dimension, int dimensionLength, float *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, double, INPUT(double *x, int *xShapeInfo, double *extraParams, double *z, int *zShapeInfo, int *dimension, int dimensionLength, double *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *extraParams, float16 *z, int *zShapeInfo, int *dimension, int dimensionLength, float16 *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))

// reduce1D
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))

// reduce3D
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))

// reduceXD
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))


namespace functions {
    namespace reduce {

            template <>
            _CUDA_H void ReduceFunction<float>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, float *x, int *xShapeInfo, float *extraParams, float *z, int *zShapeInfo, int *dimension, int dimensionLength, float *reductionBuffer, int *tadOnlyShapeInfo) {
                
                DISPATCH_SIMPLE(reduceScalarSimple, float, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, nullptr, 1, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
                
                checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <>
            _CUDA_H void ReduceFunction<float16>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, float16 *x, int *xShapeInfo, float16 *extraParams, float16 *z, int *zShapeInfo, int *dimension, int dimensionLength, float16 *reductionBuffer, int *tadOnlyShapeInfo) {
                
                DISPATCH_SIMPLE(reduceScalarSimple, float16, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, nullptr, 1, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
                
                checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <>
            _CUDA_H void ReduceFunction<double>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, double *x, int *xShapeInfo, double *extraParams, double *z, int *zShapeInfo, int *dimension, int dimensionLength, double *reductionBuffer, int *tadOnlyShapeInfo) {
                
                DISPATCH_SIMPLE(reduceScalarSimple, double, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, nullptr, 1, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
                
                checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <>
            _CUDA_H void ReduceFunction<float>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                if (rank == 1) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric1D, float, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else if (rank <= 3) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric3D, float, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else {
                    DISPATCH_SIMPLE(reduceSimpleGenericXD, float, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                }

                if (nd4j::Environment::getInstance()->isDebug())
		            checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <>
            _CUDA_H void ReduceFunction<float16>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                if (rank == 1) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric1D, float16, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else if (rank <= 3) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric3D, float16, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else {
                    DISPATCH_SIMPLE(reduceSimpleGenericXD, float16, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                }

                if (nd4j::Environment::getInstance()->isDebug())
		            checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <>
            _CUDA_H void ReduceFunction<double>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

                if (rank == 1) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric1D, double, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else if (rank <= 3) {
                    DISPATCH_SIMPLE(reduceSimpleGeneric3D, double, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                } else {
                    DISPATCH_SIMPLE(reduceSimpleGenericXD, double, PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
                }

                if (nd4j::Environment::getInstance()->isDebug())
		            checkCudaErrors(cudaStreamSynchronize(*stream));
            }

            template <typename T>
            __device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize) {
                int sPartialsLength = sMemSize / sizeof(T);
                T *sPartialsDeref = (T *) *sPartials;
                for (int i = 0; i < sPartialsLength; i++) {
                    sPartialsDeref[i] = extraParams[0];
                }
            }

            template <typename T>
            template <typename OpType>
			__device__ void ReduceFunction<T>::transformCuda1D(T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials;// = (T *)manager->getSharedReductionBuffer();
				__shared__ int tadLength;
				__shared__ int tadEWS;
				__shared__ int numTads;
				if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;
				}
				__syncthreads();

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];
					T *rX = dx + tadOffsetForBlock;

					sPartials[threadIdx.x] = OpType::startingValue(rX);

                    if (tadEWS >= 1) {
					    for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(rX[i * tadEWS], extraParams), extraParams);
					    }
					} else {
                        __shared__ int tadRank;
				        __shared__ int *tadShape;
				        __shared__ int *tadStride;
				        int xCoord[MAX_RANK];
				        if (threadIdx.x == 0) {
                            tadRank = shape::rank(tadOnlyShapeInfo);
                            tadShape = shape::shapeOf(tadOnlyShapeInfo);
                            tadStride = shape::stride(tadOnlyShapeInfo);
				        }
    				    __syncthreads();

                        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						    shape::ind2subC(tadRank, tadShape, i, xCoord);
						    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					    }
					}


					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);


					__syncthreads();
					if (threadIdx.x == 0) {
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
					}
				}
			}


            template <typename T>
            template <typename OpType>
			__device__ void ReduceFunction<T>::execScalarCuda(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
				int elementWiseStride = shape::elementWiseStride(xShapeInfo);

				int n = shape::length(xShapeInfo);

				int tid = blockDim.x * blockIdx.x + threadIdx.x;

				//shared memory space for storing intermediate results
				T *sPartials = (T *)manager->getSharedReductionBuffer();

				sPartials[threadIdx.x] = OpType::startingValue(dx);

				if (elementWiseStride >= 1) {
					for (int i = tid; i < n; i += (blockDim.x * gridDim.x)) {
						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[i * elementWiseStride], extraParams), extraParams);
					}
				}
				else {
				    __shared__ int rank;
				    __shared__ int *xShape;
				    __shared__ int *xStride;
				    if (threadIdx.x == 0) {
                        rank = shape::rank(xShapeInfo);
                        xShape = shape::shapeOf(xShapeInfo);
                        xStride = shape::stride(xShapeInfo);
				    }
				    __syncthreads();

					int ind2sub[MAX_RANK];

					for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
						shape::ind2subC(rank, xShape, i, ind2sub);

						Nd4jIndex offset = shape::getOffset(0, xShape, xStride, ind2sub, rank);
						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[offset], extraParams), extraParams);
					}
				}

				__syncthreads();
				aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), extraParams);


				__syncthreads();

				if (gridDim.x > 1) {
					unsigned int *tc = (unsigned int *)reductionBuffer;
					__shared__ bool amLast;
					tid = threadIdx.x;
					if (threadIdx.x == 0) {
						reductionBuffer[blockIdx.x] = sPartials[0];//this->postProcess(sPartials[0],n,extraParams);
					}
					__threadfence();
					__syncthreads();

					if (threadIdx.x == 0) {
						unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
						amLast = (ticket == gridDim.x - 1);
					}

					__syncthreads();

					if (amLast) {
						tc[16384] = 0;

						sPartials[threadIdx.x] = OpType::startingValue(dx);

						for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
							sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraParams);
						}
						__syncthreads();



						aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraParams);

						__syncthreads();
						if (threadIdx.x == 0) {
							result[0] = OpType::postProcess(sPartials[0], n, extraParams);
						}
					}
				}
				else {
					if (threadIdx.x == 0) {
						unsigned int *tc = (unsigned *)reductionBuffer;
						tc[16384] = 0;
						result[0] = OpType::postProcess(sPartials[0], n, extraParams);
					}
				}
			}

            template <typename T>
            template <typename OpType>
			__device__ void ReduceFunction<T>::transformCuda3D(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials; // = (T *)manager->getSharedReductionBuffer();

				__shared__ int tadLength;
				__shared__ int tadRank;
				__shared__ int numTads;
				__shared__ int *tadShape;
				__shared__ int *tadStride;
				if (threadIdx.x == 0) {
				    extern __shared__ unsigned char shmem[];
				    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadRank = shape::rank(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;

					tadShape = shape::shapeOf(tadOnlyShapeInfo);
					tadStride = shape::stride(tadOnlyShapeInfo);
				}
				__syncthreads();

				int xCoord[3];

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];

					sPartials[threadIdx.x] = OpType::startingValue(dx + tadOffsetForBlock);

					for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						shape::ind2subC(tadRank, tadShape, i, xCoord);
						Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					}
					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

					__syncthreads();
					if (threadIdx.x == 0)
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
				}
			}

            template <typename T>
            template <typename OpType>
			__device__ void ReduceFunction<T>::transformCudaXD(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo,
				Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials;

				//                __shared__ shape::TAD *tad;
				__shared__ int tadLength;
				__shared__ int tadRank;
				__shared__ int numTads;
				__shared__ int *tadShape;
				__shared__ int *tadStride;
				if (threadIdx.x == 0) {
				    extern __shared__ unsigned char shmem[];
				    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadRank = shape::rank(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;

					tadShape = shape::shapeOf(tadOnlyShapeInfo);
					tadStride = shape::stride(tadOnlyShapeInfo);
				}
				__syncthreads();

				int xCoord[MAX_RANK];

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];

					sPartials[threadIdx.x] = OpType::startingValue(dx + tadOffsetForBlock);

					for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						shape::ind2subC(tadRank, tadShape, i, xCoord);
						Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					}
					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);


					__syncthreads();
					if (threadIdx.x == 0)
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
				}
			}

            template <typename T>
            template <typename OpType>
			__device__ void ReduceFunction<T>::aggregatePartials(T *sPartials, int tid, int numItems, T *extraParams) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				int floorPow2 = numItems;

				if (floorPow2 & (floorPow2 - 1)) {
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}
					if (tid >= floorPow2) {
						sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
					}

					__syncthreads();
				}


				for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads && tid + activeThreads < numItems) {
						sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
					}
                    __syncthreads();
				}
			}


		template void ReduceFunction<float>::aggregatePartials<simdOps::Max<float> >(float*, int, int, float*);
		template void ReduceFunction<double>::aggregatePartials<simdOps::Max<double> >(double*, int, int, double*);
		template void ReduceFunction<float16>::aggregatePartials<simdOps::Max<float16> >(float16*, int, int, float16*);

		template void ReduceFunction<float>::execScalarCuda<simdOps::Max<float> >(float*, int*, float*, float*, int*, float*, UnifiedSharedMemory *, int*);
		template void ReduceFunction<double>::execScalarCuda<simdOps::Max<double> >(double*, int*, double*, double*, int*, double*, UnifiedSharedMemory *, int*);
		template void ReduceFunction<float16>::execScalarCuda<simdOps::Max<float16> >(float16*, int*, float16*, float16*, int*, float16*, UnifiedSharedMemory *, int*);

		template void ReduceFunction<float>::execScalarCuda<simdOps::Sum<float> >(float*, int*, float*, float*, int*, float*, UnifiedSharedMemory *, int*);
		template void ReduceFunction<double>::execScalarCuda<simdOps::Sum<double> >(double*, int*, double*, double*, int*, double*, UnifiedSharedMemory *, int*);
		template void ReduceFunction<float16>::execScalarCuda<simdOps::Sum<float16> >(float16*, int*, float16*, float16*, int*, float16*, UnifiedSharedMemory *, int*);
    }
}