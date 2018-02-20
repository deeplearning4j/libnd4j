//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NODE_PROFILE_H
#define LIBND4J_NODE_PROFILE_H

#include <pointercast.h>
#include <dll.h>
#include <string>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT NodeProfile {
        private:
            int _id;
            std::string _name;

            // time spent during deserialization
            Nd4jIndex _buildTime = 0L;
            
            // time spent before op execution
            Nd4jIndex _preparationTime = 0L;

            // time spent for op execution
            Nd4jIndex _executionTime = 0L;

            // amount of memory used for outputs
            Nd4jIndex _memoryActivations = 0L;

            // amount of memory used internally for temporary arrays
            Nd4jIndex _memoryTemporary = 0L;

            // amount of memory used internally for objects
            Nd4jIndex _memoryObjects = 0L;
        public:
            NodeProfile() = default;
            ~NodeProfile() = default;

            explicit NodeProfile(int id, const char *name);
        };
    }
}

#endif