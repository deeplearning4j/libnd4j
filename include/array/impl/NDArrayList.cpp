//
// @author raver119@gmail.com
//

#include <NDArrayFactory.h>
#include <array/NDArrayList.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {

    template <typename T>
    NDArrayList<T>::NDArrayList(bool expandable) {
        _expandable = expandable;
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
    void NDArrayList<T>::unstack(NDArray<T>* array, int axis) {
        _axis = axis;
        std::vector<int> args({axis});
        std::vector<int> newAxis = ShapeUtils<T>::convertAxisToTadTarget(array->rankOf(), args);
        auto result = nd4j::NDArrayFactory<T>::allTensorsAlongDimension(array, newAxis);
        for (int e = 0; e < result->size(); e++) {
            auto chunk = result->at(e)->dup(array->ordering());
            write(e, chunk);
        }
        delete result;
    }

    template <typename T>
    NDArray<T>* NDArrayList<T>::stack() {
        // FIXME: this is bad for perf, but ok as poc
        nd4j::ops::stack<T> op;
        std::vector<NDArray<T>*> inputs;
        std::vector<T> targs;
        std::vector<int> iargs;

        for (int e = 0; e < _elements.load(); e++)
            inputs.emplace_back(_chunks[e]);

        iargs.push_back(_axis);

        auto result = op.execute(inputs, targs, iargs);

        auto array = result->at(0)->dup(result->at(0)->ordering());

        delete result;

        return array;
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

    template <typename T>
    NDArrayList<T>* NDArrayList<T>::clone() {
        NDArrayList<T>* list = new NDArrayList<T>(_expandable);
        list->_axis = _axis;
        list->_id = _id;
        list->_name = _name;
        list->_elements.store(_elements.load());

        return list;
    }

    template class ND4J_EXPORT NDArrayList<float>;
    template class ND4J_EXPORT NDArrayList<float16>;
    template class ND4J_EXPORT NDArrayList<double>;
}