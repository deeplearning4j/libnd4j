//
// Created by raver119 on 06.02.2018.
//

#ifndef LIBND4J_FRAMESTATE_H
#define LIBND4J_FRAMESTATE_H

#include <string>

namespace nd4j {
    namespace graph {
        class FrameState {
        private:
            std::string _name;
            int _id = 0;
            int _numberOfCycles = 0;
            bool _activated = false;
        public:
             FrameState(int id = 0);
            ~FrameState() = default;

            /**
             * This method returns number of cycles passed for this Frame
             *
             * @return
             */
            int getNumberOfCycles();

            /**
             * This method increments number of cycles by 1 for this Frame
             */
            void incrementNumberOfCycles();

            /**
             * This method returns TRUE is frame was activated at LoopCond
             * @return
             */
            bool wasActivated();

            /**
             * This method allows to toggle activated state of this Frame
             * @param reallyActivated
             */
            void markActivated(bool reallyActivated);

            /**
             * This method returns of this Frame (if it's set)
             * @return
             */
            std::string& getFrameName();
        };
    }
}


#endif //LIBND4J_FRAMESTATE_H
