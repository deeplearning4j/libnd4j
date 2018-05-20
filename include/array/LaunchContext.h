//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H

#include <memory/Workspace.h>

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#endif

using namespace nd4j;
using namespace nd4j::memory;

namespace nd4j {
        class LaunchContext {
        private:
            nd4j::memory::Workspace *_workspace = nullptr;

#ifdef __CUDACC__
            cudaStream_t *_stream;

#endif

        public:
            LaunchContext();
            ~LaunchContext() = default;


            Workspace * workspace();

#ifdef __CUDACC__
            /**
             * This method returns pointer to cudaStream designated to given LaunchContext instance
             */
            cudaStream_t* stream();
#endif
        };
}


#endif //LIBND4J_CUDACONTEXT_H
