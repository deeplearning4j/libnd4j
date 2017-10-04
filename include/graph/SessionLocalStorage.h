//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SESSIONLOCALSTORAGE_H
#define LIBND4J_SESSIONLOCALSTORAGE_H

#include "VariableSpace.h"
#include "Block.h"
#include "Stash.h"
#include "Workspace.h"

namespace nd4j{
    namespace graph {
        class SessionLocalStorage {
        protected:
            // we start from 1, since key 0 holds original VariableSpace
            std::atomic<Nd4jIndex> _sessionCounter = 1;
            std::map<Nd4jIndex, Nd4jIndex> _threadSession;

            Nd4jIndex getThreadId();


        public:
            Nd4jIndex startSession();
            void endSession(Nd4jIndex sessionId);
        };
    }
}



template <typename T>
Nd4jIndex nd4j::graph::SessionLocalStorage<T>::getThreadId() {
#ifdef __APPLE__
    // syscall?
#elif _WIN32
    // some win32api
#else
    // syscall!
#endif
    return 0;
}

template <typename T>
void nd4j::graph::SessionLocalStorage<T>::endSession(Nd4jIndex sessionId) {
    // we should delete specific holders here
}

template <typename T>
Nd4jIndex nd4j::graph::SessionLocalStorage<T>::startSession() {
    //
    return _sessionCounter++;
}

#endif //LIBND4J_SESSIONLOCALSTORAGE_H
