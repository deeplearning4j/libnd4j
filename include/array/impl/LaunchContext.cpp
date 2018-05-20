//
// Created by raver119 on 30.11.17.
//

#include <array/LaunchContext.h>

namespace nd4j {
    LaunchContext::LaunchContext() {
        // default constructor, just to make clang/ranlib happy
    }

    Workspace* LaunchContext::workspace() {
        return _workspace;
    }
}