#pragma once

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <memory>


namespace cutw
{

namespace detail
{

void host_allocate(void*& data, std::size_t bytes);

void host_free(void* data);

}

template<typename T>
class HostArray
{
public:
    static std::shared_ptr<HostArray> create(T* const data, const std::size_t size, const bool pinned = false)
    {
        return std::shared_ptr<HostArray>{new HostArray{data, size, pinned}};
    }

    static std::shared_ptr<HostArray> create(const std::size_t size)
    {
        return std::shared_ptr<HostArray>{new HostArray{size}};
    }

    ~HostArray()
    {
        if (pinned_)
        {
            detail::host_free(data_);
        }
        else
        {
            delete[] data_;
        }
    }

    HostArray(const HostArray&) = delete;
    HostArray& operator=(const HostArray&) = delete;
    HostArray(HostArray&&) = delete;
    HostArray& operator=(HostArray&&) = delete;

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
    HostArray(T* const data, const std::size_t size, const bool pinned = false)
        : pinned_{pinned}
        , data_{data}
        , size_{size}
    {
        assert(data_);
        assert(size_ > 0);
    }

    explicit
    HostArray(const std::size_t size)
        : pinned_{true}
        , size_{size}
    {
        assert(size_ > 0);
        detail::host_allocate(reinterpret_cast<void*&>(data_), size_ * sizeof(T));
    }

    bool pinned_ = false;
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}
