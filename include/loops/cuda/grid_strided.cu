


#include <op_boilerplate.h>
#include <helpers/TAD.h>
#include <loops/scalar.h>
#include <loops/pairwise_transform.h>
#include "../grid.h"
#include <types/float16.h>


#define GRID_WIDTH 19 // number of pointers within single grid row

/*
template <typename T>
__device__ inline static void metaPredicateReduceGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                         T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, T *reductionBuffer, T *extraA, T *extraB, T scalarA, T scalarB, bool scalarReturned) {

    __shared__ UnifiedSharedMemory *manager;
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

        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }
    __syncthreads();

    // this op can be used only for reduce calls now

    if (opTypeA == 0) { // scalar

        //   DISPATCH_METAOP(functions::reduce::ReduceFunction<T>::template transformCuda6D, PARAMS(dx, xShapeInfo, paramsPtr, dz, zShapeInfo,  dimension, dimensionLength, reductionBuffer, manager, tadShapeInfo, tadOffsets), ReduceMetaOp, OPS_A(SCALAR_OPS), OPS_B(REDUCE_OPS));
    } else if (opTypeA == 1) { // transform

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("Unknown opTypeA: [%i]\n", opTypeA);
    }
}
*/

template <typename T>
__device__ inline static void metaPredicateStridedGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                          long N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB
) {
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
#ifdef __EXPERIMENTAL__
    if (opTypeB == 0) { // SCALAR
        if (opTypeA == 0) {
            // double scalar
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(SCALAR_OPS));
        } else if (opTypeA == 1) {
            // transform
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        } else if (opTypeA == 2) {
            // pwt
            // this is the most important thing here: its Dup() + Scalar
            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    } else if (opTypeB == 1) { // TRANSFORM
        if (opTypeA == 0) {
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(TRANSFORM_OPS));
        }
    } else if (opTypeB == 2) { // PWT
        if (opTypeA == 0) { // SCALAR

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 1) { // TRANSFORM

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(TRANSFORM_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 2) {

        }
    } else {
        if (threadIdx.x == 0 && blockIdx.x)
            printf("Unknown opTypeB: [%i]\n", opTypeB);
    }
#else
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            //      DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
            //      functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::InvertedMetaOp<T, simdOps::Copy<T>, simdOps::Multiply<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr);
        }
    }
#endif
}

template<typename T, typename OpClass>
__device__ static inline void invertedMetaPairwiseStridedGeneric(const int opTypeA, const int opTypeB, long N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB) {
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

    functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<OpClass>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr);
};


// kernels set for pairwise + scalar based on stride                                                                                         const int opTypeA, const int opTypeB, long N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB
DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, double, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, float16, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))



namespace functions {
    namespace grid {

        template <>
        void GRID<float>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
                    DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
                }
            }
        }

        template <>
        void GRID<float16>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
                    DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), float16, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
                }
            }
        }

        template <>
        void GRID<double>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
            if (opTypeA == 2) {
                if (opTypeB == 0) {
                    DISPATCH_METAOP(invertedMetaPairwiseStrided_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB), double, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
                }
            }
        }

        //template class GRID<float>;
        //template class GRID<float16>;
        //template class GRID<double>;
    }
}