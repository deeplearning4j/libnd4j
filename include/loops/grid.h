//
// @author raver119@gmail.com
//
//
#ifndef LIBND4J_GRID_H
#define LIBND4J_GRID_H

#include <ops/ops.h>
#include <types/float16.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace functions {
    namespace grid {
        template <typename T>
        class GRID {
        public:
            static void execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB);

            static void execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB);

            template<typename OpType>
            static __device__ void transformCuda(T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);

            template<typename OpType>
            static __device__ void transformCuda(Nd4jIndex n, T *dx, T *dy, int incx, int incy, T *params, T *result, int incz,int *allocationPointer, UnifiedSharedMemory *manager,int *tadOnlyShapeInfo);
        };
    }
}

#endif //LIBND4J_GRID_H
