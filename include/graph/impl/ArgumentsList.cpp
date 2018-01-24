//
// Created by raver119 on 24.01.18.
//

#include <graph/ArgumentsList.h>

namespace nd4j {
namespace graph {
    int ArgumentsList::size() {
        return (int) _arguments.size();
    }

    Pair&  ArgumentsList::at(int index) {
        return _arguments.at(index);
    }
}
}
