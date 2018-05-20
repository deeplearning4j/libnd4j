//
// Created by raver on 5/19/2018.
//

#ifndef LIBND4J_ND4J_H
#define LIBND4J_ND4J_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <initializer_list>
#include <vector>
#include "NDArray.h"
#include <dll.h>
#include <memory/Workspace.h>
#include <array/LaunchContext.h>

class Nd4j {
private:
    Nd4j() = default;
    ~Nd4j() = default;
public:
    template <typename T>
    NDArray<T>* create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {}) {
        //LaunchContext::segment(0);
        return nullptr;
    }

    template <typename T>
    NDArray<T>* create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {}) {
        //LaunchContext::segment(0);
        return nullptr;
    }

    template <typename T>
    NDArray<T>* createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {}) {
        return nullptr;
    }

    template <typename T>
    NDArray<T>* createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {}) {
        return nullptr;
    }
};


#endif //LIBND4J_ND4J_H
