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

    Nd4jLong LaunchContext::deviceId() {
        return _deviceId;
    }
    nd4j::random::RandomBuffer* LaunchContext::rng() {
        return _rng;
    }


    LaunchContext* LaunchContext::setDeviceId(int deviceId) {
        _deviceId = deviceId;
        return this;
    }
    LaunchContext* LaunchContext::setWorkspace(Workspace *workspace) {
        _workspace = workspace;
        return this;
    }

    LaunchContext* LaunchContext::setRng(nd4j::random::RandomBuffer *rng) {
        _rng = rng;
        return this;
    }

    LaunchContext* LaunchContext::defaultContext() {
        /**
         * defaultContext should be platform-specific
         */
        return nullptr;
    }
}