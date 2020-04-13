#pragma once

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <memory>

namespace cutw
{

namespace detail
{

void device_allocate(void*& data, std::size_t bytes);

void device_free(void* data);

}

template<typename T>
class DeviceArray
{
public:
    static std::shared_ptr<DeviceArray> create(T* const data, const std::size_t size)
    {
        return std::shared_ptr<DeviceArray>{new DeviceArray{data, size}};
    }

    static std::shared_ptr<DeviceArray> create(const std::size_t size)
    {
        return std::shared_ptr<DeviceArray>{new DeviceArray{size}};
    }

    ~DeviceArray()
    {
        detail::device_free(data_);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&&) = delete;
    DeviceArray& operator=(DeviceArray&&) = delete;

    const T* data() const
    {
        return data_;
    }

    T* data()
    {
        return data_;
    }

    std::size_t size() const
    {
        return size_;
    }

private:
    DeviceArray(T* const data, const std::size_t size)
        : data_{data}
        , size_{size}
    {
        assert(data_);
        assert(size_ > 0);
    }

    explicit
    DeviceArray(const std::size_t size)
        : size_{size}
    {
        assert(size_ > 0);
        detail::device_allocate(reinterpret_cast<void*&>(data_), size_ * sizeof(T));
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}
