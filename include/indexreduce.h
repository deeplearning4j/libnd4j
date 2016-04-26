/*
 * indexreduce.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include <shape.h>
#include <op.h>
#include <omp.h>
#include <dll.h>

#ifdef __CUDACC__
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define MAX_FLOAT 1e37
#define MIN_FLOAT 1e-37
#ifdef __JNI__
#include <jni.h>
#endif
#include <pairwise_util.h>

namespace functions {
	namespace indexreduce {
		template<typename T>
		struct IndexValue {
			T value;
			int index;
		};

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
		class IndexReduce: public  functions::ops::Op<T> {

		public:
			/**
             *
             * @param val
             * @param extraParams
             * @return
             */
			//an op for the kernel
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			IndexValue<T> op(IndexValue<T> val, T *extraParams) = 0;

			/**
             *
             * @param old
             * @param opOutput
             * @param extraParams
             * @return
             */
			//calculate an update of the reduce operation
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			IndexValue<T> update(IndexValue<T> old, IndexValue<T> opOutput,
								 T *extraParams) = 0;

			/**
             *
             * @param f1
             * @param f2
             * @param extraParams
             * @return
             */
			//invoked when combining two kernels
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			IndexValue<T> merge(IndexValue<T> f1, IndexValue<T> f2, T *extraParams) = 0;

			/**
             *
             * @param reduction
             * @param n
             * @param xOffset
             * @param dx
             * @param incx
             * @param extraParams
             * @param result
             * @return
             */
			//post process result (for things like means etc)
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			IndexValue<T> postProcess(IndexValue<T> reduction, int n, int xOffset,
									  T *dx, int incx, T *extraParams, T *result) = 0;

			/**
             *
             * @param d1
             * @param d2
             * @param extraParams
             * @return
             */
			virtual
#ifdef __CUDACC__
			inline __host__  __device__

#elif defined(__GNUC__)


#endif
			IndexValue<T> op(IndexValue<T> d1, IndexValue<T> d2, T *extraParams) = 0;

#ifdef __CUDACC__
			/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
	virtual __device__ void aggregatePartials(IndexValue<T> **sPartialsRef,int tid,int numElements,T *extraParams) {
		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		IndexValue<T> *sPartials = *sPartialsRef;
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
			while ( floorPow2 & (floorPow2 - 1) ) {
				floorPow2 &= floorPow2 - 1;
			}

			if (tid >= floorPow2) {
				IndexValue<T> prev = sPartials[tid - floorPow2];
				IndexValue<T> curr = sPartials[tid];
				sPartials[tid - floorPow2] = update(prev,curr,extraParams);
			}
			__syncthreads();
		}

#pragma unroll
		for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
			if (tid < activeThreads && tid + activeThreads < numElements) {
				IndexValue<T> curr = sPartials[tid];
				IndexValue<T> next = sPartials[tid + activeThreads];
				sPartials[tid] = update(curr,next,extraParams);
			}
			__syncthreads();
		}

	}

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
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeInfo,
			T *extraParams,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength,
			int postProcessOrNot, int *allocationBuffer, T *reductionBuffer) {
		/**
		 * Gpu information for the problem
		 */
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ volatile int resultScalar;

		__shared__ int xElementWiseStride;

		int numElements = blockDim.x;
		//shared memory space for storing intermediate results
		IndexValue<T> *sPartials;
		functions::indexreduce::SharedIndexValue<T> holder;

		sPartials = holder.getPointer();
		T startingVal = this->startingValue(dx);

#pragma unroll
		for (int i = threadIdx.x; i < numElements; i += blockDim.x) {
			IndexValue <T> val = {startingVal, i};
			sPartials[i] = val;
		}
		__syncthreads();

		//length for the tad
		__shared__ volatile int xLength;

		__shared__ volatile int resultLength;



		//only compute the tad indexes once
		IndexValue <T> reduction = {startingVal, 0};

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

			xElementWiseStride = shape::elementWiseStride(xShapeInfo);

			xLength = shape::length(xShapeInfo);
		}
		__syncthreads();

		if (!resultScalar) {
			if (dimensionLength > 1) {
				__shared__ int numOnes;
				__shared__ bool squeezed;
				__shared__ bool newSqueezeDimensions;
				__shared__ int *inputShapeInfo;
				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem

				__shared__ int *tadShapeShapeInfo;

				if(threadIdx.x == 0) {
					inputShapeInfo = xShapeInfo;
				}

				__syncthreads();

				int *shape = shape::shapeOf(inputShapeInfo);
				int *stride = shape::stride(inputShapeInfo);
				int wholeRank = shape::rank(inputShapeInfo);

				if(threadIdx.x == 0) {
					numOnes = 0;
					for(int i = 0; i < wholeRank; i++) {
						if(shape[i] == 1)
							numOnes++;
					}

				}

				__syncthreads();

				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				if(threadIdx.x == 0)
					tadShapeShapeInfo = shape::shapeInfoOnlyShapeAndStride(inputShapeInfo,dimension,dimensionLength,false);
				__syncthreads();

				int *xShape = shape::shapeOf(tadShapeShapeInfo);
				int *xStride = shape::stride(tadShapeShapeInfo);
				int tadLength = shape::length(tadShapeShapeInfo);
				int rank = shape::rank(tadShapeShapeInfo);

#pragma unroll
				for(int i = tid; i < resultLength; i+= gridDim.x * blockDim.x) {
					int offset = shape::tadOffset(i,inputShapeInfo,dimension,dimensionLength);
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int rankIter = rank;
					int xStridesIter[MAX_RANK];
					T *xPointer = dx + offset;
					IndexValue<T> indexValue;
					indexValue.index = 0;
					indexValue.value = dx[offset];

					if(PrepareOneRawArrayIter<T>(rankIter,
							xShape,
							xPointer,
							xStride,
							&rankIter,
							shapeIter,
							&xPointer,
							xStridesIter) >= 0) {
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							/* Process the innermost dimension */
							//start = update(start,op(xPointer[0],extraParams),extraParams);
							IndexValue<T> comp;
							comp.index = shape::sub2Ind(rank,xShape,coord);
							comp.value = xPointer[0];
							indexValue =  update(indexValue,comp,extraParams);
						} ND4J_RAW_ITER_ONE_NEXT(dim,
								rank,
								coord,
								shapeIter,
								xPointer,
								xStridesIter);
					}
					else {
						printf("Unable to prepare array\n");
					}

					result[i] = indexValue.index;
				}

				__syncthreads();
				if (threadIdx.x == 0) {
					delete[] tadShapeShapeInfo;


					if(numOnes > 0) {
						delete[] inputShapeInfo;
					}
				}
			} else {
				int tadLength = xLength / resultLength;
				__shared__ int offsetForTad;
#pragma unroll
				for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
					if (threadIdx.x == 0)
						offsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength);
					__syncthreads();
					sPartials[threadIdx.x] = {dx[offsetForTad], 0};
#pragma unroll
					for (int x = threadIdx.x; x < tadLength; x+= blockDim.x) {
						int indexX = offsetForTad + xElementWiseStride * x;
						IndexValue<T> comp {dx[indexX], x};
						sPartials[threadIdx.x] =  update(sPartials[threadIdx.x], comp, extraParams);
					}

					__syncthreads();
					aggregatePartials(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

					__syncthreads();
					if (threadIdx.x == 0) {
						result[i] = sPartials[threadIdx.x].index; //postProcess(sPartials[0],tadLength ,extraParams);
					}
				}

			/*
				__syncthreads();



				int resultLength = shape::length(resultShapeInfo);


				int elementsPerReductionIndex = shape::length(xShapeInfo) / resultLength;
				int xLength = shape::length(xShapeInfo);
				int i = 0,j = 0;

#pragma unroll
				for(i = tid; i < resultLength; i+= blockDim.x * gridDim.x) {
					int offsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength);
					IndexValue<T> comp2;
					comp2.value = dx[offsetForTad];
					comp2.index = 0;
					sPartials[threadIdx.x] = comp2;

					for(j = 1; j < elementsPerReductionIndex; j++) {
						IndexValue<T> comp;
						comp.value = dx[offsetForTad + xElementWiseStride * j];
						comp.index =  j;
						sPartials[threadIdx.x] =  update(sPartials[threadIdx.x],comp, extraParams);
					}

					result[i] = sPartials[threadIdx.x].index;
				}

				__syncthreads();
				*/
			}
		}


		//reduce to 1 result
		else if (resultScalar) {
			int n = shape::length(xShapeInfo);

			if(xElementWiseStride >= 1) {
				if(xElementWiseStride == 1) {
#pragma unroll
					for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
						IndexValue <T> indexVal = {dx[i], i};
						reduction = update(reduction, indexVal, extraParams);
					}
				} else {
#pragma unroll
					for(int i = xElementWiseStride * tid;i < n; i += (blockDim.x * gridDim.x * xElementWiseStride)) {
						IndexValue <T> indexVal = {dx[i * xElementWiseStride], i};
						reduction = update(reduction, indexVal, extraParams);
					}
				}
			} else {
				int rank = shape::rank(xShapeInfo);
				long allocSize = sizeof(int) * rank;
				int *ind2sub = shape::cuMalloc(allocationBuffer, allocSize);
#pragma unroll
				for(int i = tid;i < n; i += blockDim.x * gridDim.x) {
					shape::ind2sub(rank,shape::shapeOf(xShapeInfo),i,ind2sub);
					int offset = shape::getOffset(0,xShapeInfo,shape::stride(xShapeInfo),ind2sub,rank);
					int currIdx = i;
					IndexValue <T> indexVal = {dx[offset], currIdx};
					reduction = update(reduction, indexVal, extraParams);
				}

				if (tid * allocSize > PREALLOC_SIZE - allocSize) {
                	delete[] ind2sub;
            	}
			}


			sPartials[threadIdx.x] = reduction;
			__syncthreads();

			aggregatePartials(&sPartials, threadIdx.x, blockDim.x,extraParams);
			__syncthreads();

			if (gridDim.x > 1) {
				__shared__ bool amLast;
				unsigned int *tc = (unsigned *) reductionBuffer;
				int rank = shape::rank(xShapeInfo);
				tid = threadIdx.x;
				if (threadIdx.x == 0) {
					IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;
					pBuffer[blockIdx.x] = {sPartials[0].value, sPartials[0].index};
				}
				__syncthreads();
				__threadfence();

				if (tid==0) {
					unsigned int ticket = atomicInc(&tc[4096], gridDim.x);
				    amLast = (ticket == gridDim.x-1);
				}

				__syncthreads();

				if (amLast) {
					tc[4096] = 0;
					IndexValue<T> *pBuffer = (IndexValue<T> *) reductionBuffer;

					if (threadIdx.x < gridDim.x)
						sPartials[threadIdx.x] =  pBuffer[threadIdx.x];


					__syncthreads();
					aggregatePartials(&sPartials, threadIdx.x,gridDim.x,extraParams);

					__syncthreads();
					if (tid == 0) {
						result[0] = sPartials[0].index;
					}
				}
			} else {
				if (tid == 0) {
					unsigned int *tc = (unsigned *) reductionBuffer;
					tc[4096] = 0;
					result[0] = sPartials[0].index;
				}
			}
		}
	}


#endif
			/**
             * CPU operations
             * @param x the input data
             * @param xShapeInfo the shape information for the input data
             * @param extraParams the extra parameters
             * @param result the result data
             * @param resultShapeInfo the shpae information
             */
			virtual
#ifdef __CUDACC__
			inline __host__

#elif defined(__GNUC__)

#endif
			T execScalar(T *x,
						 int *xShapeInfo,
						 T *extraParams) {

				T startingVal = this->startingValue(x);
				IndexValue<T> startingIndex;
				startingIndex.value = startingVal;
				startingIndex.index = 0;
				int length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				if(xElementWiseStride < 1) {
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];

					int *xShape = shape::shapeOf(xShapeInfo);
					int *xStride = shape::stride(xShapeInfo);
					int rank = shape::rank(xShapeInfo);
					if(PrepareOneRawArrayIter<T>(rank,
												 xShape,
												 x,
												 xStride,
												 &rank,
												 shapeIter,
												 &x,
												 xStridesIter) >= 0) {

						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
							/* Process the innermost dimension */
							int i = shape::getOffset(0,xShape,xStride,coord,rank);
							IndexValue<T> curr;
							curr.value = x[i];
							curr.index = i;
							startingIndex = update(startingIndex, curr,
												   extraParams);
						} ND4J_RAW_ITER_ONE_NEXT(dim,
												 rank,
												 coord,
												 shapeIter,
												 x,
												 xStridesIter);
						return startingIndex.index;
					}
					else {
						printf("Unable to prepare array\n");
					}

				}
				else {

					if (xElementWiseStride == 1) {
						if(length < 8000) {
#pragma omp simd
							for (int i = 0; i < length; i++) {
								IndexValue<T> curr;
								curr.value = x[i];
								curr.index = i;
								startingIndex = update(startingIndex, curr,
													   extraParams);



							}
							return startingIndex.index;
						}
						else {
							BlockInformation info(length);

#pragma omp parallel

							{
								IndexValue<T> local;
								local.value = this->startingValue(x);
								local.index = 0;

								for (int i = omp_get_thread_num(); i < info.chunks; i+= info.threads) {
									int newOffset = (i * info.items);
									T *chunk = x + newOffset;
									int itemsToLoop = info.items;
									if(newOffset >= length) {
										break;
									}

									//handle modulo case
									if(newOffset + info.items >= length) {
										itemsToLoop = length - newOffset;
									}

									for (int j = 0; j < itemsToLoop; j++) {
										IndexValue<T> curr;
										curr.value = chunk[j];
										curr.index = j;
										local = update(local, curr, extraParams);
									}


#pragma omp critical
									{
										startingIndex = update(startingIndex, local,
															   extraParams);
									}


								}
							}

							return startingIndex.index;
						}

					}

					else {
#pragma omp parallel for
						for (int i = 0; i < length; i++) {
							IndexValue<T> curr;
							curr.value = x[i * xElementWiseStride];
							curr.index = i;
#pragma omp critical
							{
								startingIndex = update(startingIndex, curr,
													   extraParams);
							}

						}



					}


				}

				return  startingIndex.index;

			}


			/**
             * CPU operations
             * @param x the input data
             * @param xShapeInfo the shape information for the input data
             * @param extraParams the extra parameters
             * @param result the result data
             * @param resultShapeInfo the shpae information
             */
			virtual
#ifdef __CUDACC__
			inline __host__

#elif defined(__GNUC__)

#endif
			void exec(T *x,
					  int *xShapeInfo,
					  T *extraParams,
					  T *result,
					  int *resultShapeInfo) {
				result[0] = this->execScalar(x,xShapeInfo,extraParams);
			}

			/**
             * The dimension wise
             * CPU implementation
             * @param x the input data
             * @param xShapeInfo the x shape information
             * @param extraParams the extra parameters for the reduce
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to do reduce along long
             * @param dimensionLength the length of the dimension
             * buffer
             */
			virtual
#ifdef __CUDACC__
			inline __host__

#elif defined(__GNUC__)

#endif
			void exec(T *x,
					  int *xShapeInfo,
					  T *extraParams,
					  T *result,
					  int *resultShapeInfoBuffer,
					  int *dimension,
					  int dimensionLength) {

				if(shape::isScalar(resultShapeInfoBuffer)) {
					result[0] = execScalar(x,xShapeInfo,extraParams);
					return;
				}


				int resultLength = shape::length(resultShapeInfoBuffer);
				IndexValue<T> *startingIndex = new IndexValue<T>[resultLength];

#pragma omp parallel for
				for (int i = 0; i < resultLength; i++) {
					IndexValue<T> val;
					val.value = this->startingValue(x);
					val.index = 0;
					startingIndex[i] = val;
				}


				if(dimensionLength > 1) {
					/**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */
					int *tadShapeShapeInfo = shape::shapeInfoOnlyShapeAndStride(xShapeInfo,dimension,dimensionLength,false);
					int *xShape = shape::shapeOf(tadShapeShapeInfo);
					int *xStride = shape::stride(tadShapeShapeInfo);
					int rank = shape::rank(tadShapeShapeInfo);
#pragma omp  parallel  for
					for(int i = 0; i < resultLength; i++) {
						int offset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);
						int shapeIter[MAX_RANK];
						int coord[MAX_RANK];
						int dim;
						int rankIter = rank;
						int xStridesIter[MAX_RANK];
						T *xPointer = x + offset;
						IndexValue<T> indexValue;
						indexValue.index = 0;
						indexValue.value = x[offset];
						if(PrepareOneRawArrayIter<T>(rankIter,
													 xShape,
													 xPointer,
													 xStride,
													 &rankIter,
													 shapeIter,
													 &xPointer,
													 xStridesIter) >= 0) {
							ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
								/* Process the innermost dimension */
								IndexValue<T> comp;
								comp.index = shape::sub2Ind(rank,xShape,coord);
								comp.value = xPointer[0];
								indexValue =  update(indexValue,comp,extraParams);
							} ND4J_RAW_ITER_ONE_NEXT(dim,
													 rank,
													 coord,
													 shapeIter,
													 xPointer,
													 xStridesIter);
						}
						else {
							printf("Unable to prepare array\n");
						}



						result[i] = indexValue.index;

					}
				}

				else {
					int tadElementWiseStride = shape::tadElementWiseStride(xShapeInfo, dimension, dimensionLength);
					int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
#pragma omp parallel for
					for(int i = 0;  i < resultLength; i++) {
						int baseOffset = shape::tadOffset(i,xShapeInfo,dimension,dimensionLength);
						IndexValue<T> indexValue;
						indexValue.index = 0;
						indexValue.value = x[baseOffset];
						for(int j = 1; j < tadLength; j++) {
							IndexValue<T> comp;
							comp.index = j;
							comp.value = x[baseOffset + tadElementWiseStride * j];
							indexValue =  update(indexValue,comp,extraParams);
						}

						result[i] = indexValue.index;
					}



				}

				delete[] startingIndex;
			}

			virtual inline
#ifdef __CUDACC__
			__host__ __device__
#endif
			void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
				//no extra params aggregation needs to happen
			}


			virtual
#ifdef __CUDACC__
			__host__ __device__
#endif
			T startingValue(T *input) = 0;


#ifdef __CUDACC__
			__host__ __device__
#elif defined(__GNUC__)

#endif
			virtual ~IndexReduce() {
			}
#ifdef __CUDACC__
			__host__ __device__
#elif defined(__GNUC__)

#endif
			IndexReduce() {
			}

		};

		namespace ops {

/**
 * Find the max index
 */
			template<typename T>
			class IMax: public  functions::indexreduce::IndexReduce<T> {
			public:

				/**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
				//an op for the kernel
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> op(
						functions::indexreduce::IndexValue<T> val, T *extraParams) override {
					return val;
				}

				/**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
				//calculate an update of the reduce operation
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> update(
						functions::indexreduce::IndexValue<T> old,
						functions::indexreduce::IndexValue<T> opOutput, T *extraParams) override {
					if (opOutput.value > old.value)
						return opOutput;

#ifdef __CUDACC__
					// workaround for cuda race condition at merge phase
		else if (opOutput.value == old.value && opOutput.index < old.index)
			return opOutput;
#elif defined(__GNUC__)

#endif
					return old;
				}

				/**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
				//invoked when combining two kernels
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> merge(
						functions::indexreduce::IndexValue<T> f1,
						functions::indexreduce::IndexValue<T> f2, T *extraParams) override {
					if (f1.value > f2.value)
						return f2;
					return f1;
				}

				/**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
				//post process result (for things like means etc)
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> postProcess(
						functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
						T *dx, int incx, T *extraParams, T *result) override {
					return reduction;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(T *input) {
					return MIN_FLOAT;
				}

				/**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
								 functions::indexreduce::IndexValue<T> d2, T *extraParams) override {
					return d1;
				}
#ifdef __CUDACC__
				__host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~IMax() {
				}
#ifdef __CUDACC__
				__host__ __device__
#elif defined(__GNUC__)

#endif
				IMax() {
				}

			};

/**
 * Find the min index
 */
			template<typename T>
			class IMin: public  functions::indexreduce::IndexReduce<T> {
			public:

				/**
                 *
                 * @param val
                 * @param extraParams
                 * @return
                 */
				//an op for the kernel
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> op(
						functions::indexreduce::IndexValue<T> val, T *extraParams) override {
					return val;
				}
				virtual
#ifdef __CUDACC__
				__host__ __device__
#endif
				T startingValue(T *input) {
					return MAX_FLOAT;
				}
				/**
                 *
                 * @param old
                 * @param opOutput
                 * @param extraParams
                 * @return
                 */
				//calculate an update of the reduce operation
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> update(
						functions::indexreduce::IndexValue<T> old,
						functions::indexreduce::IndexValue<T> opOutput, T *extraParams) override {
					if (opOutput.value < old.value)
						return opOutput;

#ifdef __CUDACC__
					// workaround for cuda race condition at merge phase
		 else if (opOutput.value == old.value && opOutput.index < old.index)
			return opOutput;
#elif defined(__GNUC__)

#endif
					return old;
				}

				/**
                 *
                 * @param f1
                 * @param f2
                 * @param extraParams
                 * @return
                 */
				//invoked when combining two kernels
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> merge(
						functions::indexreduce::IndexValue<T> f1,
						functions::indexreduce::IndexValue<T> f2, T *extraParams) override {
					if (f1.value < f2.value)
						return f2;
					return f1;
				}

				/**
                 *
                 * @param reduction
                 * @param n
                 * @param xOffset
                 * @param dx
                 * @param incx
                 * @param extraParams
                 * @param result
                 * @return
                 */
				//post process result (for things like means etc)
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				functions::indexreduce::IndexValue<T> postProcess(
						functions::indexreduce::IndexValue<T> reduction, int n, int xOffset,
						T *dx, int incx, T *extraParams, T *result) override {
					return reduction;
				}

				/**
                 *
                 * @param d1
                 * @param d2
                 * @param extraParams
                 * @return
                 */
				virtual
#ifdef __CUDACC__
				inline __host__  __device__

#elif defined(__GNUC__)


#endif
				IndexValue<T> op(functions::indexreduce::IndexValue<T> d1,
								 functions::indexreduce::IndexValue<T> d2, T *extraParams) override {
					return d1;
				}

#ifdef __CUDACC__
				__host__ __device__
#elif defined(__GNUC__)

#endif
				virtual ~IMin() {
				}
#ifdef __CUDACC__
				__host__ __device__
#elif defined(__GNUC__)

#endif
				IMin() {
				}
			};
		}

		template<typename T>
		class IndexReduceOpFactory {
		public:

#ifdef __CUDACC__
			__host__ __device__
#endif
			IndexReduceOpFactory() {
			}


#ifdef __CUDACC__
			__inline__ __device__
            functions::indexreduce::IndexReduce<T> * getOp(int op, unsigned char *buffer) {
#else
			functions::indexreduce::IndexReduce<T> * getOp(int op) {
#endif
				if (op == 0) {
#ifdef __CUDACC__
					return new(buffer) functions::indexreduce::ops::IMax<T>();
#else
					return new functions::indexreduce::ops::IMax<T>();
#endif
				} else if (op == 1) {
#ifdef __CUDACC__
					return new(buffer) functions::indexreduce::ops::IMin<T>();
#else
					return new functions::indexreduce::ops::IMin<T>();
#endif
				}
				return nullptr;
			}
		};
	}


}


#ifdef __CUDACC__

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T>
__device__ void indexReduceGeneric(
		int op,
		T *dx,
		int *xShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationBuffer, T *reductionBuffer) {

	__shared__ unsigned char  __align__(8) factoryBuffer[sizeof(functions::indexreduce::IndexReduceOpFactory<T>)];
	__shared__ unsigned char  __align__(8) functionBuffer[sizeof(functions::indexreduce::IndexReduce<T>)];

	__shared__ functions::indexreduce::IndexReduce<T> *indexReduce;
	__shared__ functions::indexreduce::IndexReduceOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0) {
		newOpFactory = new(factoryBuffer) functions::indexreduce::IndexReduceOpFactory<T>();
		indexReduce = newOpFactory->getOp(op, functionBuffer);
	}
	__syncthreads();

	indexReduce->transform(dx,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,postProcessOrNot, allocationBuffer, reductionBuffer);
}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
__global__ void indexReduceDouble(
		int op,
		double *dx,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationBuffer, double *reductionBuffer) {
	indexReduceGeneric<double>(
			op,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationBuffer, reductionBuffer);

}

/**
 * The external driver
 * api interface to the cuda kernel
 * @param op the operation number to execute
 * @param n the length of the input
 * @param dx the input data
 * @param xShapeInfo the input data shape information
 * @param extraParams  the extra parameters for the reduce
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the shape information for the data
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
__global__ void indexReduceFloat(
		int op,
		float *dx,
		int *xShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,  int *allocationBuffer, float *reductionBuffer) {
	indexReduceGeneric<float>(
			op,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationBuffer, reductionBuffer);

}



#endif

#endif /* INDEXREDUCE_H_ */

