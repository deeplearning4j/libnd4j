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

template <typename T>
class Nd4j {
private:
    nd4j::LaunchContext *_context;


protected:
    nd4j::LaunchContext* context();
public:
    Nd4j(LaunchContext *context = nullptr);
    ~Nd4j() = default;

    NDArray<T>* create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

    NDArray<T>* create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

    NDArray<T>* createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

    NDArray<T>* createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});
};


#endif //LIBND4J_ND4J_H
