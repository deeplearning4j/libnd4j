//
// Created by raver119 on 17.10.2017.
//

#ifndef LIBND4J_LEGACYBROADCASTOP_H
#define LIBND4J_LEGACYBROADCASTOP_H

#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class LegacyBroadcastOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Block<T>& block);
        public:
            LegacyBroadcastOp();
            LegacyBroadcastOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block);
        };
    }
}


#endif //LIBND4J_LEGACYBROADCASTOP_H
