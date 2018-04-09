//
// Created by raver on 4/9/2018.
//

#include "../indexreduce.h"
#include <op_boilerplate.h>

#include "../legacy_ops.h"

namespace functions {
    namespace indexreduce {

        template <typename T>
        T IndexReduce<T>::execScalar(
                const int opNum,
                T *x,
                int *xShapeInfo,
                T *extraParams) {
            RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
        }

        template <typename T>
        void IndexReduce<T>::exec(const int opNum,
                  T *x,
                  int *xShapeInfo,
                  T *extraParams,
                  T *result,
                  int *resultShapeInfoBuffer,
                  int *dimension,
                  int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffset) {
            DISPATCH_BY_OPNUM(exec, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
        }

        template <typename T>
        template<typename OpType>
        T IndexReduce<T>::execScalar(T *x, int *xShapeInfo, T *extraParams) {

            //T startingVal = OpType::startingValue(x);
            IndexValue<T> startingIndex = OpType::startingIndexValue(x);

            Nd4jIndex length = shape::length(xShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
            if(xElementWiseStride < 1) {
                int *xShape = shape::shapeOf(xShapeInfo);
                int *xStride = shape::stride(xShapeInfo);
                int tadRank = shape::rank(xShapeInfo);
                int xCoord[MAX_RANK];

                for (Nd4jIndex i = 0; i < length; i++) {
                    shape::ind2subC(tadRank,xShape, i, xCoord);
                    Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, tadRank);

                    IndexValue<T> curr;
                    curr.value = x[xOffset];
                    curr.index = i;

                    startingIndex = OpType::update(startingIndex, curr, extraParams);
                }
                return startingIndex.index;
            }
            else {

                if (xElementWiseStride == 1) {
                    if(length < ELEMENT_THRESHOLD) {
// FIXME: proper reduction to be used here
//#pragma omp simd
                        for (Nd4jIndex i = 0; i < length; i++) {
                            IndexValue<T> curr;
                            curr.value = x[i];
                            curr.index = i;
                            startingIndex = OpType::update(startingIndex, curr, extraParams);

                        }
                        return startingIndex.index;
                    }
                    else {
                        BlockInformation info(length, ELEMENT_THRESHOLD);

#pragma omp parallel num_threads(info.threads) if (info.threads > 1) default(shared)

                        {
                            IndexValue<T> local = OpType::startingIndexValue(x);


                            for (Nd4jIndex i = omp_get_thread_num(); i < info.chunks; i+= info.threads) {
                                Nd4jIndex newOffset = (i * info.items);
                                T *chunk = x + newOffset;
                                Nd4jIndex itemsToLoop = info.items;
                                if(newOffset >= length) {
                                    break;
                                }

                                //handle modulo case
                                if(newOffset + info.items >= length) {
                                    itemsToLoop = length - newOffset;
                                }

                                for (Nd4jIndex j = 0; j < itemsToLoop; j++) {
                                    IndexValue<T> curr;
                                    curr.value = chunk[j];
                                    curr.index = newOffset + j;
                                    local = OpType::update(local, curr, extraParams);
                                }

#pragma omp critical
                                {
                                    startingIndex = OpType::update(startingIndex, local, extraParams);
                                }
                            }
                        }

                        return startingIndex.index;
                    }

                }

                else {
                    for (Nd4jIndex i = 0; i < length; i++) {
                        IndexValue<T> curr;
                        curr.value = x[i * xElementWiseStride];
                        curr.index = i;
                        startingIndex = OpType::update(startingIndex, curr,
                                                       extraParams);
                    }
                }
            }

            return  startingIndex.index;
        };

        template <typename T>
        template<typename OpType>
        void IndexReduce<T>::exec(T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffset) {

            if(shape::isScalar(resultShapeInfoBuffer)) {
                result[0] = execScalar<OpType>(x,xShapeInfo,extraParams);
                return;
            }

            const Nd4jIndex resultLength = shape::length(resultShapeInfoBuffer);
            IndexValue<T> *startingIndex = new IndexValue<T>[resultLength];

#pragma omp parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
            for (Nd4jIndex i = 0; i < resultLength; i++) {
                IndexValue<T> val = OpType::startingIndexValue(x);
                startingIndex[i] = val;
            }

            int *tadOnlyShapeInfo = tadShapeInfo;
            Nd4jIndex *tadOffsets = tadOffset;
            shape::TAD *tad = nullptr;

            if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                tad->createTadOnlyShapeInfo();
                tad->createOffsets();

                if (tad->dimensionLength < 1) {
                    delete tad;
                    delete[] startingIndex;
                    return;
                }

                tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
                tadOffsets = tad->tadOffsets;
            }

            int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
            int numTads = shape::length(xShapeInfo) / tadLength;


            if(!(shape::elementWiseStride(tadOnlyShapeInfo) > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo)))) {
                /**
                             * The element wise stride belong longs to a reduction index.
                             * When used out of order, we can get rid of the data
                             * dependencies and rely on using the max dimension
                             * specified for stride instead.
                             * Say we take the sum(0,1) along long arr
                             * we can use arr.stride(1) as a representation
                             * along long which to iterate.
                             */

                int *tadShapeShapeInfo = tadOnlyShapeInfo;
                int *xShape = shape::shapeOf(tadShapeShapeInfo);
                int *xStride = shape::stride(tadShapeShapeInfo);
                int rank = shape::rank(tadShapeShapeInfo);


#pragma omp  parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
                for(Nd4jIndex i = 0; i < resultLength; i++) {
                    Nd4jIndex offset = tadOffsets[i];


                    IndexValue<T> indexValue = OpType::startingIndexValue(&x[offset]);

                    int xCoord[MAX_RANK];

                    for(int j = 0; j < tadLength; j++) {
                        shape::ind2subC(rank,xShape, j, xCoord);
                        Nd4jIndex xOffset = shape::getOffset(offset, xShape, xStride, xCoord, rank);

                        IndexValue<T> comp;
                        comp.index = j;
                        comp.value = x[xOffset];
                        indexValue = OpType::update(indexValue,comp,extraParams);
                    }
                    result[i] = indexValue.index;
                }
            } else {
                int tadElementWiseStride = shape::elementWiseStride(tadOnlyShapeInfo);
                //const int tadLength = shape::length(tadOnlyShapeInfo);

//#pragma omp parallel for schedule(guided) if (resultLength > TAD_THRESHOLD) default(shared)
                for(Nd4jIndex i = 0;  i < resultLength; i++) {
                    Nd4jIndex baseOffset = tadOffsets[i];
                    IndexValue<T> indexValue = OpType::startingIndexValue(&x[baseOffset]);

// FIXME: proper reduction required here
                    for(int j = 0; j < tadLength; j++) {
                        IndexValue<T> comp;
                        comp.index = j;
                        comp.value = x[baseOffset + tadElementWiseStride * j];
                        indexValue = OpType::update(indexValue,comp,extraParams);
                    }
                    result[i] = indexValue.index;
                }
            }

            delete[] startingIndex;
        }
    };
}