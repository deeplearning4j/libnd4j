//
// Created by raver on 5/20/2018.
//

#ifndef LIBND4J_POINTERFACTORY_H
#define LIBND4J_POINTERFACTORY_H

#include <NDArray.h>
#include <array/LaunchContext.h>
#include <initializer_list>
#include <vector>
#include <pointercast.h>


namespace nd4j {
    template <typename T>
    class PointerFactory {
    private:
        LaunchContext *_context;

    public:
        PointerFactory() = default;
        ~PointerFactory() = default;

        LaunchContext* context();

        NDArray<T>* valueOf(std::initializer_list<Nd4jLong> shape, T value, char order = 'c');

        NDArray<T>* create(std::initializer_list<Nd4jLong> shape, char order = 'c', std::initializer_list<T> data = {});

        NDArray<T>* createUninitialized(std::initializer_list<Nd4jLong> shape, char order = 'c');
    };
}

#endif //LIBND4J_POINTERFACTORY_H
