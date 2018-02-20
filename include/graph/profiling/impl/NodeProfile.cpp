//
//  @author raver119@gmail.com
//

#include <graph/profiling/NodeProfile.h>

namespace nd4j {
    namespace graph {
        NodeProfile::NodeProfile(int id, const char *name) {
            _id = id;
            _name = name;
        };
    }
}