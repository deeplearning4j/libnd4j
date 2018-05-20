//
// @author raver119@gmail.com
//

#include "../Nd4j.h"

using namespace nd4j;

template <typename T>
Nd4j<T>::Nd4j(LaunchContext *context) {
    this->_context = context;
}

template <typename T>
LaunchContext* Nd4j<T>::context() {
    return _context;
}

template <typename T>
NDArray<T>* Nd4j<T>::create(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
   return nullptr;
}


template <typename T>
NDArray<T>* Nd4j<T>::create(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}


template <typename T>
NDArray<T>* Nd4j<T>::createUninitialized(char order, std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}

template <typename T>
NDArray<T>* Nd4j<T>::createUninitialized(std::initializer_list<Nd4jLong> shape, std::initializer_list<T> data) {
    return nullptr;
}
