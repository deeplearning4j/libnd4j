//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_HELPERS_H
#define LIBND4J_HELPERS_H

#include <graph/Context.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _im2col(nd4j::graph::Context<T>& context, T *dst, T *src, int *outShape, int *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode);
        }
    }
}

#endif //LIBND4J_HELPERS_H
