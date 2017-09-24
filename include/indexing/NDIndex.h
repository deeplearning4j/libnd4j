//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <vector>

namespace nd4j {
    class NDIndex {
    protected:
        std::vector<int> _indices;
    public:
        NDIndex() {
            //
        }

        ~NDIndex() {
            //
        }

        std::vector<int>& getIndices();
    };

    class NDIndexAll : public NDIndex {
    public:
        NDIndexAll() : nd4j::NDIndex() {
            _indices.push_back(-1);
        }

        ~NDIndexAll() {
            //
        }
    };


    class NDIndexPoint : public NDIndex {
    public:
        explicit NDIndexPoint(int point): nd4j::NDIndex() {
            this->_indices.push_back(point);
        }

        ~NDIndexPoint() {
            //
        };
    };

    class NDIndexInterval : public NDIndex {
    public:
        explicit NDIndexInterval(int start, int end): nd4j::NDIndex() {
            for (int e = start; e < end; e++)
                this->_indices.push_back(e);
        }


        ~NDIndexInterval() {
            //
        }
    };
}

std::vector<int>& nd4j::NDIndex::getIndices() {
    return _indices;
}

#endif //LIBND4J_NDINDEX_H
