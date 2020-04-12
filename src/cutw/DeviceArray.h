#pragma once

#include <cassert>
#include <algorithm>
#include <cstddef>


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
    DeviceArray(T* const data, const std::size_t size)
        : data_{data}
        , size_{size}
    {
        assert(data_);
        assert(size_ > 0);
    }

    DeviceArray(const std::size_t size)
        : size_{size}
    {
        assert(size_ > 0);
        detail::device_allocate(reinterpret_cast<void*&>(data_), size_ * sizeof(T));
    }

    ~DeviceArray()
    {
        free();
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& o)
    {
        swap(o);
    }

    DeviceArray& operator=(DeviceArray&& o)
    {
        if (this != &o)
        {
            if (data_)
            {
                free();
                data_ = nullptr;
                size_ = 0;
            }
            swap(o);
        }
        return *this;
    }

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
    void free()
    {
        if (!data_)
        {
            return;
        }
        detail::device_free(data_);
    }

    void swap(DeviceArray& o)
    {
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}
