


#include <op_boilerplate.h>
#include <pointercast.h>
#include <helpers/TAD.h>
#include <types/float16.h>
#include "../grid_shaped.h"


#include <ops/ops.h>
#include "../legacy_ops.h"


#define GRID_WIDTH 19 // number of pointers within single grid row

template <typename T>
__device__ inline static void metaPredicateShapeGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                        long N, T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = (Nd4jPointer *) &scalarA;
        }
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) {
            params[1] = (Nd4jPointer *) &scalarB;
        }
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            //    DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
            //  functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::InvertedMetaOp<T, simdOps::Copy<T>, simdOps::Multiply<T>>>(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
        }
    }
}

template<typename T, typename OpClass>
__device__ static inline void invertedMetaPairwiseShapedGeneric(const int opTypeA, const int opTypeB, long N, T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = (Nd4jPointer *) &scalarA;
        }
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) {
            params[1] = (Nd4jPointer *) &scalarB;
        }
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();

    functions::grid::GRIDShaped<T>::template transformCuda<OpClass>(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
};



#ifndef __CLION_IDE__
// kernels set for pairwise + scalar based on shape
DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, double, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float16, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
#endif

namespace functions {
    namespace grid {

        __device__ void _ind2subC(int rank, int *shape, int idx, int *coords) {
            shape::ind2subC(rank, shape, idx, coords);
        }

        __device__ Nd4jIndex _getOffset(Nd4jIndex offset,  int *shape, int *stride, int *coords, int rank) {
            return shape::getOffset(offset, shape, stride, coords, rank);
        }

        __device__ int* _shapeOf(int *shape) {
            return shape::shapeOf(shape);
        }

        __device__ int* _stride(int *shape) {
            return shape::stride(shape);
        }

        __device__ int _rank(int* shape) {
            return shape::rank(shape);
        }

        template<typename T>
        template<typename OpType>
        __device__ void GRIDShaped<T>::transformCuda(T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ int xRank;
            __shared__ int yRank;
            __shared__ int resultRank;
            __shared__ Nd4jIndex n;

            __shared__ int *xShape;
            __shared__ int *yShape;
            __shared__ int *zShape;

            __shared__ int *xStride;
            __shared__ int *yStride;
            __shared__ int *zStride;

            if (threadIdx.x == 0) {
                xRank = _rank(xShapeBuffer);
                yRank = _rank(yShapeBuffer);
                resultRank = _rank(resultShapeBuffer);
                n = shape::length(xShapeBuffer);

                xShape = _shape(xShapeBuffer);
                yShape = _shape(yShapeBuffer);

                if (dx != result) {
                    zShape = _shape(resultShapeBuffer);
                    zStride = _stride(resultShapeBuffer);
                }

                xStride = _stride(xShapeBuffer);
                yStride = _stride(yShapeBuffer);
            }
            __syncthreads();

            int xCoord[MAX_RANK];
            int yCoord[MAX_RANK];

            if (dx == result) {
                for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, xCoord);
                    _ind2subC(yRank, yShape, i, yCoord);

                    Nd4jIndex xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    Nd4jIndex yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
                }
            } else {
                int resultCoord[MAX_RANK];

                for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, xCoord);
                    _ind2subC(yRank, yShape, i, yCoord);
                    _ind2subC(resultRank, zShape, i, resultCoord);

                    Nd4jIndex xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    Nd4jIndex yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    Nd4jIndex resultOffset = _getOffset(0, zShape, zStride, resultCoord, resultRank);
                    result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
                }
            }
        }


        template <>
        void GRIDShaped<float>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
#ifndef __CLION_IDE__
                    DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
#endif
                }
            }
        }

        template <>
        void GRIDShaped<float16>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
#ifndef __CLION_IDE__
                    DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
#endif
                }
            }
        }

        template <>
        void GRIDShaped<double>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
#ifndef __CLION_IDE__
                    DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
#endif
                }
            }
        }
    }
}