//
//  @author raver119@gmail.com
//

#include <helpers/OpTracker.h>

namespace nd4j {
    
    OpTracker* OpTracker::getInstance() {
        if (_INSTANCE == 0)
            _INSTANCE = new OpTracker();

        return _INSTANCE;
    }

    void OpTracker::storeOperation(nd4j::graph::OpType opType, const OpDescriptor& descriptor) {
        if (_map.count(opType) < 1) {
            std::vector<OpDescriptor> vec;
            _map[opType] = vec;
        }

        _operations++;

        _map[opType].emplace_back(descriptor);
    }

    void OpTracker::storeOperation(nd4j::graph::OpType opType, const char* opName, const Nd4jIndex opNum) {
        OpDescriptor descriptor(0, opName, false);
        storeOperation(opType, descriptor);
    }

    int OpTracker::totalGroups() {
        return (int) _map.size();
    }

    int OpTracker::totalOperations() {
        return _operations;
    }

    nd4j::OpTracker* nd4j::OpTracker::_INSTANCE = 0;
}