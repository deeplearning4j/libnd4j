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

#include <array/ObjectFactory.h>
#include <array/PointerFactory.h>

#include <types/float16.h>
#include <mutex>

class Nd4j {
private:
    nd4j::LaunchContext *_context;

    // meh
    static std::map<Nd4jLong, nd4j::ObjectFactory<float>> _factoriesOF;
    static std::map<Nd4jLong, nd4j::ObjectFactory<float16>> _factoriesOH;
    static std::map<Nd4jLong, nd4j::ObjectFactory<double>> _factoriesOD;

    static std::map<Nd4jLong, nd4j::PointerFactory<float>> _factoriesPF;
    static std::map<Nd4jLong, nd4j::PointerFactory<float16>> _factoriesPH;
    static std::map<Nd4jLong, nd4j::PointerFactory<double>> _factoriesPD;



    // meh 2.0
    static thread_local nd4j::ObjectFactory<float> _objectFactoryF;
    static thread_local nd4j::PointerFactory<float> _pointerFactoryF;

    static thread_local nd4j::ObjectFactory<float16> _objectFactoryH;
    static thread_local nd4j::PointerFactory<float16> _pointerFactoryH;

    static thread_local nd4j::ObjectFactory<double> _objectFactoryD;
    static thread_local nd4j::PointerFactory<double> _pointerFactoryD;


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
