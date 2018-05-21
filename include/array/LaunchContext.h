//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H

#include <memory/Workspace.h>
#include <helpers/helper_random.h>

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
            // memory workspace for this context
            nd4j::memory::Workspace *_workspace = nullptr;

            // RNG instance used in this context
            nd4j::random::RandomBuffer *_rng = nullptr;

            // id of computational device
            Nd4jLong _deviceId = 0;

#ifdef __CUDACC__
            // cuda stream that will be used for this context
            cudaStream_t *_stream;

            // cublas?
#endif

        public:
            LaunchContext();
            ~LaunchContext() = default;

            /////////////////////////
            Workspace* workspace();
            Nd4jLong deviceId();
            nd4j::random::RandomBuffer* rng();


            LaunchContext* setDeviceId(int deviceId);
            LaunchContext* setWorkspace(Workspace *workspace);
            LaunchContext* setRng(nd4j::random::RandomBuffer *rng);

#ifdef __CUDACC__
            /**
             * This method returns pointer to cudaStream designated to given LaunchContext instance
             */
            cudaStream_t* stream();

            LaunchContext* setCudaStream(cudaStream_t *stream);
#endif

            static LaunchContext* defaultContext();
        };
}


#endif //LIBND4J_CUDACONTEXT_H
