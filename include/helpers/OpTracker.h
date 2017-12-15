//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_OP_TRACKER_H
#define LIBND4J_OP_TRACKER_H

#include <map>
#include <vector>
#include <pointercast.h>
#include <graph/generated/utils_generated.h>
#include <ops/declarable/OpDescriptor.h>

using namespace nd4j::ops;
using namespace nd4j::graph;

namespace nd4j {
    class OpTracker {
    private:
        static OpTracker* _INSTANCE;        

        std::map<OpType, std::vector<OpDescriptor>> _map;

        OpTracker() = default;
        ~OpTracker() = default;
    public:
        static OpTracker* getInstance();

        void storeOperation(nd4j::graph::OpType opType, const OpDescriptor& descriptor);
        void storeOperation(nd4j::graph::OpType opType, const char* opName, const Nd4jIndex opNum);
    };
}

#endif