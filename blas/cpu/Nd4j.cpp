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
    } else
        return nullptr;
}

template <>
PointerFactory<float16>* Nd4j::p<float16>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_pointerFactoryH;
    } else
        return nullptr;
}


template <>
PointerFactory<double>* Nd4j::p<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_pointerFactoryD;
    } else
        return nullptr;
}


template <>
nd4j::ObjectFactory<float>* Nd4j::o<float>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryF;
    } else
        return nullptr;
}

template <>
nd4j::ObjectFactory<float16>* Nd4j::o<float16>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryH;
    } else
        return nullptr;
}

template <>
nd4j::ObjectFactory<double>* Nd4j::o<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        return &_objectFactoryD;
    } else
        return nullptr;
}

///////////////


thread_local nd4j::ObjectFactory<float> Nd4j::_objectFactoryF;
thread_local nd4j::PointerFactory<float> Nd4j::_pointerFactoryF;

thread_local nd4j::ObjectFactory<float16> Nd4j::_objectFactoryH;
thread_local nd4j::PointerFactory<float16> Nd4j::_pointerFactoryH;

thread_local nd4j::ObjectFactory<double> Nd4j::_objectFactoryD;
thread_local nd4j::PointerFactory<double> Nd4j::_pointerFactoryD;



template nd4j::ObjectFactory<float>* Nd4j::o<float>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<float16>* Nd4j::o<float16>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<double>* Nd4j::o<double>(nd4j::LaunchContext *);

template nd4j::PointerFactory<float>* Nd4j::p<float>(nd4j::LaunchContext *);
template nd4j::PointerFactory<float16>* Nd4j::p<float16>(nd4j::LaunchContext *);
template nd4j::PointerFactory<double>* Nd4j::p<double>(nd4j::LaunchContext *);
