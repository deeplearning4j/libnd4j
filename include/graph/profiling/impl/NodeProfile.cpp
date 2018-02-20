//
//  @author raver119@gmail.com
//

#include <helpers/logger.h>
#include <graph/profiling/NodeProfile.h>

namespace nd4j {
    namespace graph {
        NodeProfile::NodeProfile(int id, const char *name) {
            _id = id;
            _name = name;
        };

        void NodeProfile::printOut() {
            nd4j_printf("Node: <%i/%s>\n", _id, _name.c_str());
            nd4j_printf("      Memory: ACT: %lld; TMP: %lld; OBJ: %lld;\n", _memoryActivations, _memoryTemporary, _memoryObjects);
            nd4j_printf("      Time: PREP: %lld; EXEC: %lld;\n", _preparationTime, _executionTime);
        }
    }
}