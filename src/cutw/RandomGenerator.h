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
    static std::shared_ptr<RandomGenerator> create(const std::size_t seed)
    {
        return std::shared_ptr<RandomGenerator>{new RandomGenerator{seed}};
    }

    ~RandomGenerator();

    template<typename T>
    void generateUniform(Stream& stream, DeviceArray<T>& device)
    {
        generateUniform(stream, device.data(), device.size());
    }

    curandGenerator_st* get() const;

private:
    RandomGenerator(std::size_t seed);
    void generateUniform(Stream& stream, float* device, std::size_t n);
    void generateUniform(Stream& stream, double* device, std::size_t n);
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
