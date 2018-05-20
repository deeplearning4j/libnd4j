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


namespace nd4j {
    template <typename T>
    class PointerFactory {
    public:
        nd4j::LaunchContext* context();

        NDArray<T>* create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T>* create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T>* createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T>* createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});
    };


    template <typename T>
    class ObjectFactory {
    public:
        // this method returns
        nd4j::LaunchContext* context();

        NDArray<T> create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T> create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T> createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});

        NDArray<T> createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data = {});
    };
}

class Nd4j {
private:
    nd4j::LaunchContext *_context;

public:
    Nd4j(LaunchContext *context = nullptr);
    ~Nd4j() = default;

    nd4j::LaunchContext* context();

    template <typename T>
    static PointerFactory<T>* p(LaunchContext *ctx = nullptr);

    template <typename T>
    static ObjectFactory<T>* o(LaunchContext *ctx = nullptr);
};


#endif //LIBND4J_ND4J_H
