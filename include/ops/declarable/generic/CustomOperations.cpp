//
// This is special snowflake. This file builds bindings for ops availability tests
//
// @author raver119@gmail.com
//

#include <loops/legacy_ops.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    struct _loader {
        _loader() {
            // 
            OpTracker::getInstance();
        }   
    }


    static loader;
}