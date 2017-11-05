//
// Created by raver119 on 13.10.2017.
//

#ifndef LIBND4J_BOOLEANOP_H
#define LIBND4J_BOOLEANOP_H

#include <Block.h>
#include "OpDescriptor.h"
#include "DeclarableOp.h"

namespace nd4j {
    namespace ops {
        template <typename T>
        class BooleanOp : public DeclarableOp<T> {
        protected:
            OpDescriptor * _descriptor;
        public:
            BooleanOp(const char *name, int numInputs, bool scalar);
            ~BooleanOp();

            bool evaluate(std::initializer_list<nd4j::NDArray<T> *> args);
            bool evaluate(std::vector<nd4j::NDArray<T> *>& args);
            bool evaluate(nd4j::graph::Block<T>& block);

            Nd4jStatus execute(Block<T>* block);

            ShapeList *calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) override;

        protected:
            bool prepareOutputs(Block<T>& block);
            virtual Nd4jStatus validateAndExecute(Block<T> &block) = 0;
        };
    }
}



#endif //LIBND4J_BOOLEANOP_H