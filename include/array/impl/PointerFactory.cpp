//
// Created by raver on 5/20/2018.
//

#include <array/PointerFactory.h>


namespace nd4j {

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


    template class PointerFactory<float>;
    template class PointerFactory<float16>;
    template class PointerFactory<double>;
}