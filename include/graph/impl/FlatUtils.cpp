//
// Created by raver119 on 22.11.2017.
//

#include <graph/FlatUtils.h>


namespace nd4j {
    namespace graph {
        std::pair<int, int> FlatUtils::fromIntPair(IntPair *pair) {
            return std::pair<int, int>(pair->first(), pair->second());
        }

        std::pair<Nd4jIndex, Nd4jIndex> FlatUtils::fromLongPair(LongPair *pair) {
            return std::pair<Nd4jIndex, Nd4jIndex>(pair->first(), pair->second());
        }
    }
}