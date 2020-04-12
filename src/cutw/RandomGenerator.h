#pragma once

#include <memory>

#include "DeviceArray.h"
#include "Stream.h"

struct curandGenerator_st;

namespace cutw
{

class RandomGenerator
{
public:
    RandomGenerator(Stream& stream, std::size_t seed);
    ~RandomGenerator();

    template<typename T>
    void generateUniform(DeviceArray<T>& device)
    {
        generateUniform(device.data(), device.size());
    }

    curandGenerator_st* get() const;

private:
    void generateUniform(float* device, std::size_t n);
    void generateUniform(double* device, std::size_t n);
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
