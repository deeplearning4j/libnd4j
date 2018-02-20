//
//  @author raver119@gmail.com
//

#ifndef ND4J_GRAPH_PROFILE_H
#define ND4J_GRAPH_PROFILE_H

#include "NodeProfile.h"
#include <pointercast.h>
#include <dll.h>
#include <vector>
#include <map>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT GraphProfile {
        private:
            /**
             * This is global memory values
             */
            Nd4jIndex _memoryTotal = 0L;
            Nd4jIndex _memoryActivations = 0L;
            Nd4jIndex _memoryTemporary = 0L;
            Nd4jIndex _memoryObjects = 0L;

            // time spent for graph construction
            Nd4jIndex _buildTime = 0L;

            std::vector<NodeProfile *> _profiles;
            std::map<int, NodeProfile *> _profilesById;
        public:
            GraphProfile() = default;
            ~GraphProfile();
        };
    }
}

#endif