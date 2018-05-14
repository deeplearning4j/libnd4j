//
// Created by raver119 on 13/05/18.
//

#include <system/HardwareStatus.h>

namespace nd4j {
    const char * HardwareStatus::getDeviceName() {
        return _deviceName.c_str();
    }

    const char * HardwareStatus::getDeviceDescription() {
        return _deviceDescription.c_str();
    }

    Nd4jIndex HardwareStatus::getMemoryUsed() {
        return _memoryUsed;
    }

    Nd4jIndex HardwareStatus::getMemoryTotal() {
        return _memoryTotal;
    }

    Nd4jIndex HardwareStatus::getCurrentTemperature() {
        return _temperature;
    }

    void HardwareStatus::setDeviceName(std::string& name) {
        _deviceName = name;
    }

    void HardwareStatus::setDeviceDescription(std::string& desc) {
        _deviceDescription = desc;
    }

    void HardwareStatus::setDeviceName(const char *name) {
        _deviceName = std::string(name);
    }

    void HardwareStatus::setDeviceDescription(const char *desc) {
        _deviceDescription = std::string(desc);
    }

    void HardwareStatus::setMemoryUsed(Nd4jIndex bytes) {
        _memoryUsed = bytes;
    }

    void HardwareStatus::setMemoryTotal(Nd4jIndex bytes) {
        _memoryTotal = bytes;
    }

    void HardwareStatus::setCurrentTemperature(Nd4jIndex bytes) {
        _temperature = bytes;
    }

    HardwareStatus::HardwareStatus(const HardwareStatus &other) {
        _deviceName = other._deviceName;
        _deviceDescription = other._deviceDescription;
        _memoryTotal = other._memoryTotal;
        _memoryUsed = other._memoryUsed;
    }
}
