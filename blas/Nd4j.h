//
// Created by raver on 5/19/2018.
//

#ifndef LIBND4J_ND4J_H
#define LIBND4J_ND4J_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <initializer_list>
#include <vector>
#include <map>
#include "NDArray.h"
#include <dll.h>
#include <memory/Workspace.h>
#include <array/LaunchContext.h>
#include <types/float16.h>
#include <mutex>



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


    template <typename T>
    class ObjectFactory {
    private:
        LaunchContext *_context;

    public:
        ObjectFactory() = default;
        ~ObjectFactory() = default;


        // this method returns context
        LaunchContext* context();

        NDArray<T> valueOf(std::initializer_list<Nd4jLong> shape, T value, char order = 'c');

        NDArray<T> create(std::initializer_list<Nd4jLong> shape, char order = 'c', std::initializer_list<T> data = {});

        NDArray<T> createUninitialized(std::initializer_list<Nd4jLong> shape, char order = 'c');
    };
}

class Nd4j {
private:
    nd4j::LaunchContext *_context;

    // meh
    static std::map<Nd4jLong, nd4j::ObjectFactory<float>> _factoriesOF;
    static std::map<Nd4jLong, nd4j::ObjectFactory<float16>> _factoriesOH;
    static std::map<Nd4jLong, nd4j::ObjectFactory<double >> _factoriesOD;


    static thread_local nd4j::ObjectFactory<float> _objectFactoryF;
    static thread_local nd4j::PointerFactory<float> _pointerFactoryF;


public:
    explicit Nd4j(LaunchContext *context = nullptr);
    ~Nd4j() = default;

    nd4j::LaunchContext* context();

    template <typename T>
    static nd4j::PointerFactory<T>* p(LaunchContext *ctx = nullptr);

    template <typename T>
    static nd4j::ObjectFactory<T>* o(LaunchContext *ctx = nullptr);
};


#endif //LIBND4J_ND4J_H
