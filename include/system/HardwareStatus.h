//
// Created by raver119 on 13/05/18.
//

#ifndef LIBND4J_UTILIZATIONSTATUS_H
#define LIBND4J_UTILIZATIONSTATUS_H

#include <pointercast.h>
#include <dll.h>
#include <vector>
#include <string>

namespace nd4j {
    class ND4J_EXPORT HardwareStatus {
    private:
        std::string _deviceName = "Unknown device";
        std::string _deviceDescription;

        // field for utilization. 0 stands for absolutely idle device, 100 stands for absolutely busy device.
        unsigned int _utilization = 0;

        // field for used memory amount, if applicable for given device
        Nd4jIndex _memoryUsed = 0L;

        // field for total (max) memory amount, if applicable for given device
        Nd4jIndex _memoryTotal = 0L;
    public:
        HardwareStatus() = default;
        HardwareStatus(const HardwareStatus &other);

        const char * getDeviceName();
        const char * getDeviceDescription();

        Nd4jIndex getMemoryUsed();
        Nd4jIndex getMemoryTotal();
    };
}



#endif //LIBND4J_UTILIZATIONSTATUS_H
