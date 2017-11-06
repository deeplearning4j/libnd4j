//
// This class describes collection of NDArrays
//
// @author raver119!gmail.com
//

#ifndef NDARRAY_LIST_H
#define NDARRAY_LIST_H

#include <string>
#include <atomic>
#include <map>
#include <memory/Workspace.h>
#include <NDArray.h>

namespace nd4j {
    template <typename T>
    class NDArrayList {
    private:
        // workspace where chunks belong to
        nd4j::memory::Workspace* _workspace = nullptr;

        // numeric and symbolic ids of this list
        std::pair<int, int> _id;
        std::string _name;

        // stored chunks
        std::map<int, nd4j::NDArray<T>*> _chunks;

        // just a counter, for stored elements
        std::atomic<int> _elements;

        // reference shape
        std::vector<int> _shape;

        // unstack axis
        int _axis = 0;

        //
        bool _expandable = false;

        // maximum number of elements
        int _height = 0;
    public:
        NDArrayList(int height, bool expandable = false);
        ~NDArrayList();

        NDArray<T>* read(int idx);
        Nd4jStatus write(int idx, NDArray<T>* array);

        NDArray<T>* pick(std::initializer_list<int> indices);
        NDArray<T>* pick(std::vector<int>& indices);

        NDArray<T>* stack();
        void unstack(NDArray<T>* array, int axis);

        std::pair<int,int>& id();
        std::string& name();
        nd4j::memory::Workspace* workspace();

        NDArrayList<T>* clone();

        int elements();
        int height();
    };
}

#endif