//
// Created by raver119 on 13/05/18.
//

#ifndef LIBND4J_UTILIZATIONSTATUS_H
#define LIBND4J_UTILIZATIONSTATUS_H

#include <system/HardwareFeatures.h>
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

        // field for current temperature
        Nd4jIndex _temperature = 0L;

        // bit-encoded features are stored here.
        // TODO: if one day we get above 64 distinct features - we'll just start using arrays/lists/vectors instead
        Nd4jIndex _features;
    public:
        HardwareStatus() = default;
        HardwareStatus(const HardwareStatus &other);

        ///// setters

        void setDeviceName(std::string& name);
        void setDeviceName(const char *name);

        void setDeviceDescription(std::string& desc);
        void setDeviceDescription(const char *desc);

        void setMemoryUsed(Nd4jIndex bytes);
        void setMemoryTotal(Nd4jIndex bytes);

        void setCurrentTemperature(Nd4jIndex bytes);

        ///// getters

        const char * getDeviceName();
        const char * getDeviceDescription();

        Nd4jIndex getMemoryUsed();
        Nd4jIndex getMemoryTotal();

        Nd4jIndex getCurrentTemperature();
    };
}



#endif //LIBND4J_UTILIZATIONSTATUS_H
