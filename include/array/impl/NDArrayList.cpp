//
// @author raver119@gmail.com
//

#include <array/NDArrayList.h>

namespace nd4j {

    template <typename T>
    NDArrayList<T>::NDArrayList(bool expandable) {
        _elements.store(0);
    }

    template <typename T>
    NDArrayList<T>::~NDArrayList() {
        //
    }

    template <typename T>
    NDArray<T>* NDArrayList<T>::read(int idx) {
        if (_chunks.count(idx) < 1) {
            nd4j_printf("Non-existent chunk requested: [%i]\n", idx);
            throw "Bad index";
        }

        return _chunks[idx];
    }

    template <typename T>
    Nd4jStatus NDArrayList<T>::write(int idx, NDArray<T>* array) {
        if (_chunks.count(idx) > 0)
            return ND4J_STATUS_DOUBLE_WRITE;

        // we store reference shape on first write
        if (_elements.load() == 0) {
            for (int e = 0; e < array->rankOf(); e++)
                _shape.emplace_back(array->sizeAt(e));
        } else {
            if (array->rankOf() != _shape.size())
                return ND4J_STATUS_BAD_DIMENSIONS;

            for (int e = 0; e < array->rankOf(); e++)
                if (_shape[e] != array->sizeAt(e))
                    return ND4J_STATUS_BAD_DIMENSIONS;
        }
        
        _elements++;

        // storing reference
        _chunks[idx] = array;

        return ND4J_STATUS_OK;
    }

    template <typename T>
    NDArray<T>* NDArrayList<T>::stack() {
        return nullptr;
    }

    template <typename T>
    std::pair<int,int>& NDArrayList<T>::id() {
        return _id;
    }

    template <typename T>
    std::string& NDArrayList<T>::name() {
        return _name;
    }

    template <typename T>
    nd4j::memory::Workspace* NDArrayList<T>::workspace() {
        return _workspace;
    }

    template <typename T>
    int NDArrayList<T>::elements() {
        return _elements.load();
    }

    template class ND4J_EXPORT NDArrayList<float>;
    template class ND4J_EXPORT NDArrayList<float16>;
    template class ND4J_EXPORT NDArrayList<double>;
}