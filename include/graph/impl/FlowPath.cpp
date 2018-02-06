//
// Created by raver119 on 16/11/17.
//

#include <graph/FlowPath.h>

namespace nd4j {
    namespace graph {

        void FlowPath::ensureNode(int nodeId) {
            if (_states.count(nodeId) == 0) {
                NodeState state(nodeId);
                _states[nodeId] = state;
            }
        }

        void FlowPath::ensureFrame(int frameId) {
            if (_frames.count(frameId) == 0) {
                FrameState state(frameId);
                _frames[frameId] = state;
            }
        }

        void FlowPath::setInnerTime(int nodeId, Nd4jIndex time) {
            ensureNode(nodeId);

            _states[nodeId].setInnerTime(time);
        }

        void FlowPath::setOuterTime(int nodeId, Nd4jIndex time) {
            ensureNode(nodeId);

            _states[nodeId].setOuterTime(time);
        }

        Nd4jIndex FlowPath::innerTime(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].innerTime();
        }

        Nd4jIndex FlowPath::outerTime(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].outerTime();
        }

        bool FlowPath::isNodeActive(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].isActive();
        }
            
        void FlowPath::markNodeActive(int nodeId, bool isActive) {
            ensureNode(nodeId);

            _states[nodeId].markActive(isActive);
        }

        int FlowPath::branch(int nodeId){
            ensureNode(nodeId);

            return _states[nodeId].branch();
        }

        void FlowPath::markBranch(int nodeId, int index) {
            ensureNode(nodeId);

            _states[nodeId].markBranch(index);
        }

        bool FlowPath::isFrameActive(int frameId) {
            ensureFrame(frameId);

            return _frames[frameId].wasActivated();
        }

        void FlowPath::markFrameActive(int frameId, bool isActive) {
            ensureFrame(frameId);

            _frames[frameId].markActivated(isActive);
        }
    }
}