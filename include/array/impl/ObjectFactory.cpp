//
// Created by raver on 5/20/2018.
//

#include <array/ObjectFactory.h>


namespace nd4j {
    template <typename T>
    ObjectFactory<T>::ObjectFactory(LaunchContext *context) {
        this->_context = context;
    }

    template <typename T>
    LaunchContext* ObjectFactory<T>::context() {
        return _context;
    }

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

    template class nd4j::ObjectFactory<float>;
    template class nd4j::ObjectFactory<float16>;
    template class nd4j::ObjectFactory<double>;
}