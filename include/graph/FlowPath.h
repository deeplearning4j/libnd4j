//
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_FLOWPATH_H
#define LIBND4J_FLOWPATH_H

#include <map>
#include <pointercast.h>
#include <graph/NodeState.h>
#include <graph/FrameState.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT FlowPath {
        private:
            std::map<int, NodeState> _states;
            std::map<int, FrameState> _frames;

            void ensureNode(int nodeId);
            void ensureFrame(int nodeId);
        public:
            FlowPath() = default;
            ~FlowPath() = default;

            void setInnerTime(int nodeId, Nd4jIndex time);
            void setOuterTime(int nodeId, Nd4jIndex time);

            Nd4jIndex innerTime(int nodeId);
            Nd4jIndex outerTime(int nodeId);

            bool isNodeActive(int nodeId);
            void markNodeActive(int nodeId, bool isActive);

            int branch(int nodeId);
            void markBranch(int nodeId, int index);

            // Frame-related methods

            bool isFrameActive(int frameId);
            void markFrameActive(int frameId, bool isActive);

            bool isResetPlanned(int frameId);
            void planRewind(int frameId, bool reallyRewind);

            int getRewindPosition(int frameId);
            void setRewindPositionOnce(int frameId, int position);
        };
    }
}


#endif //LIBND4J_FLOWPATH_H
