//
// @author raver119@gmail.com
//

#include "../Nd4j.h"

using namespace nd4j;


Nd4j::Nd4j(LaunchContext *context) {
    this->_context = context;
}

LaunchContext* Nd4j::context() {
    return _context;
}

template <>
PointerFactory<float>* Nd4j::p<float>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
            return &_pointerFactoryF;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesPF.count(id) == 0)
            _factoriesPF[id] = PointerFactory<float>(ctx);

        return &(_factoriesPF[id]);
    }
}

template <>
PointerFactory<float16>* Nd4j::p<float16>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_pointerFactoryH;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesPH.count(id) == 0)
            _factoriesPH[id] = PointerFactory<float16>(ctx);

        return &(_factoriesPH[id]);
    }
}


template <>
PointerFactory<double>* Nd4j::p<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_pointerFactoryD;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesPD.count(id) == 0)
            _factoriesPD[id] = PointerFactory<double>(ctx);

        return &(_factoriesPD[id]);
    }
}


template <>
nd4j::ObjectFactory<float>* Nd4j::o<float>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryF;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesOF.count(id) == 0)
            _factoriesOF[id] = ObjectFactory<float>(ctx);

        return &(_factoriesOF[id]);
    }
}

template <>
nd4j::ObjectFactory<float16>* Nd4j::o<float16>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryH;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesOH.count(id) == 0)
            _factoriesOH[id] = ObjectFactory<float16>(ctx);

        return &(_factoriesOH[id]);
    }
}

template <>
nd4j::ObjectFactory<double>* Nd4j::o<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryD;
    } else {
        auto id = reinterpret_cast<Nd4jLong>(ctx);
        if (_factoriesOD.count(id) == 0)
            _factoriesOD[id] = ObjectFactory<double>(ctx);

        return &(_factoriesOD[id]);
    }
}

///////////////

// init thread-local defaults
thread_local nd4j::ObjectFactory<float> Nd4j::_objectFactoryF;
thread_local nd4j::PointerFactory<float> Nd4j::_pointerFactoryF;

thread_local nd4j::ObjectFactory<float16> Nd4j::_objectFactoryH;
thread_local nd4j::PointerFactory<float16> Nd4j::_pointerFactoryH;

thread_local nd4j::ObjectFactory<double> Nd4j::_objectFactoryD;
thread_local nd4j::PointerFactory<double> Nd4j::_pointerFactoryD;

// init global holders for factories
std::map<Nd4jLong, nd4j::ObjectFactory<float>> Nd4j::_factoriesOF;
std::map<Nd4jLong, nd4j::ObjectFactory<float16>> Nd4j::_factoriesOH;
std::map<Nd4jLong, nd4j::ObjectFactory<double>> Nd4j::_factoriesOD;

std::map<Nd4jLong, nd4j::PointerFactory<float>> Nd4j::_factoriesPF;
std::map<Nd4jLong, nd4j::PointerFactory<float16>> Nd4j::_factoriesPH;
std::map<Nd4jLong, nd4j::PointerFactory<double>> Nd4j::_factoriesPD;


template nd4j::ObjectFactory<float>* Nd4j::o<float>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<float16>* Nd4j::o<float16>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<double>* Nd4j::o<double>(nd4j::LaunchContext *);

template nd4j::PointerFactory<float>* Nd4j::p<float>(nd4j::LaunchContext *);
template nd4j::PointerFactory<float16>* Nd4j::p<float16>(nd4j::LaunchContext *);
template nd4j::PointerFactory<double>* Nd4j::p<double>(nd4j::LaunchContext *);
