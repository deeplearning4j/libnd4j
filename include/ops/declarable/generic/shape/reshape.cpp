//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // here iArgs is vector with shape dimensions at the beginning and last element in iArgs is order
        CUSTOM_OP_IMPL(reshape, 1, 1, true, 0, -1) {
            std::vector<int>* argumets = block.getIArguments();
            int argsSize = argumets->size();

            REQUIRE_TRUE(argsSize >= 3, 0, "Reshape arguments should have order and at least 2 dimensions");

            NDArray<T> *x = INPUT_VARIABLE(0);

            char order = (char)(*argumets)[0];
            if (order != 'c' && order != 'f')
                order = x->ordering();

            std::vector<int> shapeNew;

            for (int e = 1; e < (int) argumets->size(); e++)
                shapeNew.push_back((int) argumets->at(e));

            //nd4j::Logger::printv("shapeNew: ", shapeNew);

            if(block.isInplace()) {
                if (x->reshapei(order, shapeNew)) {
                    STORE_RESULT(*x);
//                    x->printShapeInfo("New shape");
                    return ND4J_STATUS_OK;
                }
            }
            else {
                auto ret = new NDArray<T>(*x);
                if (ret->reshapei(order, shapeNew)) {
                    STORE_RESULT(*ret);
//                    ret->printShapeInfo("New shape");
                    return ND4J_STATUS_OK;
                }
            }
            return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SHAPE_FN(reshape) {
            int *inp = inputShape->at(0);

            std::vector<int>* arguments = block.getIArguments();

            char order = (char)(*arguments)[0];
            if (order != 'c' && order != 'f')
                order = shape::order(inp);

            std::vector<int> shapeNew;

            for (int e = 1; e < (int) arguments->size(); e++)
                shapeNew.push_back((int) arguments->at(e));

            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) shapeNew.size()), int);

            int numberNegativesOnes = 0;

            int* shape_ = shapeNew.data();
            for (int i = 0; i < (int) shapeNew.size(); i++) {
                if (shapeNew[i] < 0) {
                    if (numberNegativesOnes >= 1)
                        throw "Only one dimension can be negative ones";

                    numberNegativesOnes++;

                    int shapeLength = 1;
                    for (int j = 0; j < (int) shapeNew.size(); j++)
                        if (shape_[j] >= 1)
                            shapeLength *= shape_[j];

                    // FIXME: use workspace here
                    int realShape = nd4j::math::nd4j_abs<int>((int) shape::length(inp) / shapeLength);
                    int* thisNewShape = new int[shapeNew.size()];

                    for (int j = 0; j < (int) shapeNew.size(); j++) {
                        if (i != j) {
                            thisNewShape[j] = shape_[j];
                        } else
                            thisNewShape[j] = realShape;
                    }

                    shape_ = thisNewShape;
                    break;
                }
            }

            for (int e = 0; e < (int) shapeNew.size(); e++) {
                shapeNew[e] = shape_[e];
            }

            if (numberNegativesOnes > 0)
                delete[] shape_;

            newShape[0] = shapeNew.size();
            int cnt = 1;
            for (auto v: shapeNew)
                newShape[cnt++] = v;

            shape::updateStrides(newShape, order);

            return new ShapeList(newShape);
        }
    }
}