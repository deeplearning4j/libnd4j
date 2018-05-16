//
// Created by raver119 on 13/05/18.
//

#include <system/HardwareReport.h>

#include <pointercast.h>
#include <dll.h>
#include <stdexcept>

namespace nd4j {
    int HardwareReport::getNumberOfProcessors() {
        return 0;
    }

    int HardwareReport::getNumberOfCores() {
        return 0;
    }

    int HardwareReport::getNumberOfDevices() {
        return static_cast<int>(_devices.size());
    }

    HardwareStatus HardwareReport::getDeviceStatus(int deviceId) {
        if (deviceId >= _devices.size() || deviceId < 0)
            throw std::runtime_error("Requested deviceId is above total number of devices");

        return _devices[deviceId];
    }

    HardwareReport::HardwareReport(const HardwareReport &other) {
        for (auto &v: other._devices) {
            auto r = v.second;
            _devices[v.first] = r;
        }

        for (auto &v: other._processors) {
            auto r = v.second;
            _processors[v.first] = r;
        }

        for (auto &v: other._cores) {
            auto r = v.second;
            _cores[v.first] = r;
        }
    }

    void HardwareReport::storeDeviceStatus(int deviceId, HardwareStatus status) {
        if (deviceId > 0)
            if (_devices.count(deviceId - 1) == 0)
                throw std::runtime_error("All beans should be stored sequentially: 0 -> X");

        if (deviceId < 0)
            throw std::runtime_error("deviceId can't have negative value");


        _devices[deviceId] = status;
    }
}
