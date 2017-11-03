//
// @author raver119@gmail.com
//

#include <array/NDArrayList.h>

namespace nd4j {

    template <typename T>
    NDArrayList<T>::NDArrayList(bool expandable) {
        //
    }

    template <typename T>
    NDArrayList<T>::~NDArrayList() {
        //
    }

    template <typename T>
    NDArray<T>* NDArrayList<T>::read(int idx) {
        return nullptr;
    }

    template <typename T>
    void NDArrayList<T>::write(int idx, NDArray<T>* array) {

    }

    template <typename T>
    NDArray<T>* NDArrayList<T>::stack() {
        return nullptr;
    }
}