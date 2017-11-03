//
// This class describes collection of NDArrays
//
// @author raver119!gmail.com
//

#ifndef NDARRAY_LIST_H
#define NDARRAY_LIST_H

#include <NDArray.h>

namespace nd4j {
    template <typename T>
    class NDArrayList {
    public:
        NDArrayList(bool expandable = false);
        ~NDArrayList();

        NDArray<T>* read(int idx);
        void write(int idx, NDArray<T>* array);

        NDArray<T>* stack();
    };
}

#endif