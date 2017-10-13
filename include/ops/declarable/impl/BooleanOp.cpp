//
// Created by raver119 on 13.10.2017.
//

#include "ops/declarable/BooleanOp.h"
#include <vector>
#include <initializer_list>

namespace nd4j {
    namespace ops {
        template <typename T>
        BooleanOp<T>::BooleanOp(const char *name, int numInputs, bool scalar) : DeclarableOp<T>::DeclarableOp(name, numInputs, scalar) {
            //
        }

        template <typename T>
        BooleanOp<T>::~BooleanOp() {
            //
        }

        template <typename T>
        ShapeList *BooleanOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            int *shapeNew;
            ALLOCATE(shapeNew, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shapeNew[0] = 2;
            shapeNew[1] = 1;
            shapeNew[2] = 1;
            shapeNew[3] = 1;
            shapeNew[4] = 1;
            shapeNew[5] = 0;
            shapeNew[6] = 1;
            shapeNew[7] = 99;

            return new ShapeList(shapeNew);
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(nd4j::graph::Block<T> &block) {
            // check if scalar or not

            // validation?

            Nd4jStatus status = this->validateNonEmptyInput(block);
            if (status != ND4J_STATUS_OK) {
                nd4j_printf("Inputs should be not empty for BooleanOps","");
                throw "Bad inputs";
            }

            status = this->validateAndExecute(block);
            if (status == ND4J_STATUS_TRUE)
                return true;
            else if (status == ND4J_STATUS_FALSE)
                return false;
            else {
                nd4j_printf("Got error %i during [%s] evaluation: ", (int) status, this->getOpDescriptor()->getOpName()->c_str());
                throw "Internal error";
            }
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(std::initializer_list<nd4j::NDArray<T> *> args) {
            std::vector<nd4j::NDArray<T> *> vec(args);
            return this->evaluate(vec);
        }

        template <typename T>
        bool BooleanOp<T>::evaluate(std::vector<nd4j::NDArray<T> *> &args) {
            VariableSpace<T> variableSpace;

            int cnt = -1;
            std::vector<int> in;
            for (auto v: args) {
                auto var = new Variable<T>(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Block<T> block(1, &variableSpace, false);
            block.fillInputs(in);

            return this->evaluate(block);
        }


        template class BooleanOp<float>;
        template class BooleanOp<float16>;
        template class BooleanOp<double>;
    }
}

