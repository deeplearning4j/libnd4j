//
// Created by raver119 on 18.12.17.
//

#include <op_boilerplate.h>
#include <loops/summarystatsreduce.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>

namespace functions {
    namespace summarystats {


        template <typename T>
        T SummaryStatsReduce<T>::execScalar(const int opNum, const bool biasCorrected, T *x, int *xShapeInfo, T *extraParams) {
            RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams), SUMMARY_STATS_OPS);
        }

        template <typename T>
        void SummaryStatsReduce<T>::exec(const int opNum, const bool biasCorrected, T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength) {
            DISPATCH_BY_OPNUM(exec, PARAMS(biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength), SUMMARY_STATS_OPS);
        }

        template <typename T>
        template <typename OpType >
        T SummaryStatsReduce<T>::execScalar(const bool biasCorrected, T *x, int *xShapeInfo, T *extraParams) {
            SummaryStatsData<double> startingIndex;
            startingIndex.initialize();
            Nd4jIndex length = shape::length(xShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
            if (xElementWiseStride == 1) {
                for (Nd4jIndex i = 0; i < length; i++) {
                    SummaryStatsData<double> curr;
                    curr.initWithValue(x[i]);
                    startingIndex = update(startingIndex, curr,
                                           extraParams);
                }
                T finalVal = (T) OpType::getValue(biasCorrected, startingIndex);

                return finalVal;
            }
            else {
                int xCoords[MAX_RANK];

                int *xShape = shape::shapeOf(xShapeInfo);
                int *xStride = shape::stride(xShapeInfo);
                int xRank = shape::rank(xShapeInfo);


                for (Nd4jIndex i = 0; i < length; i++) {
                    shape::ind2subC(xRank, xShape, i, xCoords);
                    Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoords, xRank);

                    SummaryStatsData<double> curr;
                    curr.initWithValue(x[xOffset]);
                    startingIndex = update(startingIndex, curr, extraParams);
                }

                T finalVal = (T)OpType::getValue(biasCorrected, startingIndex);
                return finalVal;
            }
        }

        template <typename T>
        template <typename OpType >
        void SummaryStatsReduce<T>::exec(const bool biasCorrected, T *x, int *xShapeInfo, T *extraParams, T *result, int *resultShapeInfoBuffer, int *dimension, int dimensionLength) {
            if (shape::isScalar(resultShapeInfoBuffer)) {
                result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }


            shape::TAD tad(xShapeInfo, dimension, dimensionLength);
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            //no-op
            if (tad.dimensionLength < 1)
                return;

            int resultLength = shape::length(resultShapeInfoBuffer);
            //pre squeezed: this is for keeping the pointer to the original
            //shape information for tad offset
            //the squeezed information doesn't render the right strides for
            //tad offset
            if (resultLength == 1 || dimensionLength == shape::rank(xShapeInfo) || tad.wholeThing) {
                result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }

            if (!(shape::elementWiseStride(tad.tadOnlyShapeInfo) > 0 && (tad.numTads == 1 || shape::isVector(tad.tadOnlyShapeInfo) ||
                                                                         shape::isScalar(tad.tadOnlyShapeInfo) || tad.wholeThing)) && !(dimensionLength > 1)) {

                /**
                 * The element wise stride belong longs to a reduction index.
                 * When used out of order, we can get rid of the data
                 * dependencies and rely on using the max dimension
                 * specified for stride instead.
                 * Say we take the sum(0,1) along long arr
                 * we can use arr.stride(1) as a representation
                 * along long which to iterate.
                 */

                int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                int *xShape = shape::shapeOf(tadShapeShapeInfo);
                int *xStride = shape::stride(tadShapeShapeInfo);
                int rank = shape::rank(tadShapeShapeInfo);
#pragma omp parallel for schedule(guided) default(shared)
                for (int i = 0; i < resultLength; i++) {
                    Nd4jIndex offset = tad.tadOffsets[i];
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int rankIter = rank;
                    int xStridesIter[MAX_RANK];
                    T *xPointer = x + offset;
                    SummaryStatsData<T> comp;
                    comp.initWithValue(0.0);
                    if (PrepareOneRawArrayIter<T>(rankIter,
                                                  xShape,
                                                  xPointer,
                                                  xStride,
                                                  &rankIter,
                                                  shapeIter,
                                                  &xPointer,
                                                  xStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                SummaryStatsData<T> comp2;
                                comp2.initWithValue(xPointer[0]);
                                comp = update(comp, comp2, extraParams);
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

                    result[i] = OpType::getValue(biasCorrected, comp);
                }
            }
            else {
                if (dimensionLength == 1) {
                    int tadElementWiseStride = shape::elementWiseStride(tad.tadOnlyShapeInfo);
                    int tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        Nd4jIndex baseOffset = tad.tadOffsets[i];
                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[baseOffset]);
// FIXME: reduction to be used here
                        for (int j = 1; j < tadLength; j++) {
                            SummaryStatsData<T> comp2;
                            comp2.initWithValue(x[baseOffset + (tadElementWiseStride * j)]);
                            comp = update(comp, comp2, extraParams);
                        }

                        result[i] = OpType::getValue(biasCorrected, comp);
                    }
                } else {
                    int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                    int *tadShape = shape::shapeOf(tadShapeShapeInfo);
                    int *tadStride = shape::stride(tadShapeShapeInfo);
                    int tadRank = shape::rank(tadShapeShapeInfo);
                    int tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                    for (int r = 0; r < resultLength; r++) {
                        int xCoord[MAX_RANK];
                        Nd4jIndex tadOffsetForBlock = tad.tadOffsets[r];

                        SummaryStatsData<T> comp;
                        comp.initWithValue(x[tadOffsetForBlock]);

// FIXME: reduction should be fixed
                        for (int i = 1; i < tadLength; i ++) {
                            shape::ind2subC(tadRank, tadShape, i, xCoord);
                            Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                            SummaryStatsData <T> indexVal2;
                            indexVal2.initWithValue(x[xOffset]);

                            comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
                        }
                        result[r] = OpType::getValue(biasCorrected, comp);
                    }
                }
            }
        }


        template class ND4J_EXPORT SummaryStatsReduce<float>;
        template class ND4J_EXPORT SummaryStatsReduce<float16>;
        template class ND4J_EXPORT SummaryStatsReduce<double>;
    }
}