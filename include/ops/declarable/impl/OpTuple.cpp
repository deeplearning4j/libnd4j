//
// Created by raver119 on 11.10.2017.
//

#include "ops/declarable/OpTuple.h"

nd4j::ops::OpTuple::OpTuple(const char *opName) {
    _opName = opName;
}

nd4j::ops::OpTuple::OpTuple(const char *opName, std::initializer_list<nd4j::NDArray<float> *> &&inputs, std::initializer_list<float> &&tArgs, std::initializer_list<int> &&iArgs) {
    _opName = opName;
    _inputs = inputs;
    _iArgs = iArgs;
    _tArgs = tArgs;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::addInput(nd4j::NDArray<float> *array) {
    _inputs.emplace_back(array);
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::addOutput(nd4j::NDArray<float> *array) {
    _outputs.emplace_back(array);
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::setTArgs(std::initializer_list<float> tArgs) {
    _tArgs = tArgs;
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::setIArgs(std::initializer_list<int> iArgs) {
    _iArgs = iArgs;
    return this;
}
