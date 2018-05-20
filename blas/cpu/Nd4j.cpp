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
        //return &_pointerFactoryH;
        return nullptr;
    } else
        return nullptr;
}


template <>
PointerFactory<double>* Nd4j::p<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        //return &_pointerFactoryD;
        return nullptr;
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
        //return &_objectFactoryH;
        return nullptr;
    } else
        return nullptr;
}

template <>
nd4j::ObjectFactory<double>* Nd4j::o<double>(LaunchContext *ctx) {
    // default context used, on per-thread basis
    if (ctx == nullptr) {
        //return &_objectFactoryD;
        return nullptr;
    } else
        return nullptr;
}

///////////////

template <typename T>
LaunchContext* ObjectFactory<T>::context() {
    return _context;
}

template <typename T>
LaunchContext* PointerFactory<T>::context() {
    return _context;
}


template <typename T>
NDArray<T>* PointerFactory<T>::valueOf(std::initializer_list<Nd4jLong> shape, T value, char order) {
    auto x = new NDArray<T>(order, shape);
    x->assign(value);
    return x;
}

template <typename T>
NDArray<T>* PointerFactory<T>::create(std::initializer_list<Nd4jLong> shape, char order, std::initializer_list<T> data) {
    return new NDArray<T>(order, shape, data);
}


template <typename T>
NDArray<T>* PointerFactory<T>::createUninitialized(std::initializer_list<Nd4jLong> shape, char order) {
    return new NDArray<T>(order, shape);
}

//////////

template <typename T>
NDArray<T> ObjectFactory<T>::create(std::initializer_list<Nd4jLong> shape, char order, std::initializer_list<T> data) {
    NDArray<T> x(order, shape, data);
    return x;
}


template <typename T>
NDArray<T> ObjectFactory<T>::createUninitialized(std::initializer_list<Nd4jLong> shape, char order) {
    NDArray<T> x(order, shape);
    return x;
}

template <typename T>
NDArray<T> ObjectFactory<T>::valueOf(std::initializer_list<Nd4jLong> shape, T value, char order) {
    NDArray<T> x(order, shape);
    x.assign(value);
    return x;
}

thread_local nd4j::ObjectFactory<float> Nd4j::_objectFactoryF;
thread_local nd4j::PointerFactory<float> Nd4j::_pointerFactoryF;


template class nd4j::PointerFactory<float>;
template class nd4j::PointerFactory<float16>;
template class nd4j::PointerFactory<double>;

template class nd4j::ObjectFactory<float>;
template class nd4j::ObjectFactory<float16>;
template class nd4j::ObjectFactory<double>;

template nd4j::ObjectFactory<float>* Nd4j::o<float>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<float16>* Nd4j::o<float16>(nd4j::LaunchContext *);
template nd4j::ObjectFactory<double>* Nd4j::o<double>(nd4j::LaunchContext *);

template nd4j::PointerFactory<float>* Nd4j::p<float>(nd4j::LaunchContext *);
template nd4j::PointerFactory<float16>* Nd4j::p<float16>(nd4j::LaunchContext *);
template nd4j::PointerFactory<double>* Nd4j::p<double>(nd4j::LaunchContext *);
