#pragma once

#include <cassert>
#include <algorithm>
#include <cstddef>


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

    ~HostArray()
    {
        free();
    }

    HostArray(const HostArray&) = delete;
    HostArray& operator=(const HostArray&) = delete;

    HostArray(HostArray&& o)
    {
        swap(o);
    }

    HostArray& operator=(HostArray&& o)
    {
        if (this != &o)
        {
            if (data_)
            {
                free();
                pinned_ = false;
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
        if (pinned_)
        {
            detail::host_free(data_);
        }
        else
        {
            delete[] data_;
        }
    }

    void swap(HostArray& o)
    {
        std::swap(pinned_, o.pinned_);
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
    }

    bool pinned_ = false;
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}
