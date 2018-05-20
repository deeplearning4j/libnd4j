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

template <typename T>
NDArray<T>* PointerFactory<T>::create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
   return nullptr;
}


template <typename T>
NDArray<T>* PointerFactory<T>::create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}


template <typename T>
NDArray<T>* PointerFactory<T>::createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}

template <typename T>
NDArray<T>* PointerFactory<T>::createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}

//////////

template <typename T>
NDArray<T> ObjectFactory<T>::create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}


template <typename T>
NDArray<T> ObjectFactory<T>::create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}


template <typename T>
NDArray<T> ObjectFactory<T>::createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}

template <typename T>
NDArray<T> ObjectFactory<T>::createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}
