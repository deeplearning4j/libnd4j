//
//  @author raver119@gmail.com
//

#include <graph/profiling/GraphProfile.h>

namespace nd4j {
    namespace graph {
        GraphProfile::~GraphProfile() {
            // releasing NodeProfile pointers
            for (auto v: _profiles)
                delete v;
        }
    }
}