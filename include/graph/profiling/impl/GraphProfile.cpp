//
//  @author raver119@gmail.com
//

#include <graph/profiling/GraphProfile.h>
#include <helpers/logger.h>

namespace nd4j {
    namespace graph {
        GraphProfile::~GraphProfile() {
            // releasing NodeProfile pointers
            for (auto v: _profiles)
                delete v;
        }

        void GraphProfile::addToTotal(Nd4jIndex bytes) {
            _memoryTotal += bytes;
        }

        void GraphProfile::addToActivations(Nd4jIndex bytes) {
            _memoryActivations += bytes;
        }
        
        void GraphProfile::addToTemporary(Nd4jIndex bytes) {
            _memoryTemporary += bytes;
        }
        
        void GraphProfile::addToObjects(Nd4jIndex bytes) {
            _memoryObjects += bytes;
        }

        void GraphProfile::setBuildTime(Nd4jIndex nanos) {
            _buildTime = nanos;
        }

        void GraphProfile::setExecutionTime(Nd4jIndex nanos) {
            _executionTime = nanos;
        }

        void GraphProfile::printOut() {
            nd4j_printf("Graph report:\n", "");
            nd4j_printf("Memory:\n", "");

            nd4j_printf("Time:\n", "");
            nd4j_printf("Construction time: %lld\n", _buildTime);
            nd4j_printf("Execution time: %lld\n", _executionTime);

            nd4j_printf("Per-node reports:\n", "");
            for (auto v: _profiles)
                v->printOut();
            
        }
    }
}