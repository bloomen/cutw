#pragma once

#include <cassert>

#include "DeviceArray.h"
#include "HostArray.h"
#include "Stream.h"


namespace cutw
{

namespace detail
{

void copy_to_device_impl(const void* host, void* device, std::size_t size, Stream* s);

void copy_to_host_impl(void* host, const void* device, std::size_t size, Stream* s);

}

template<typename T>
void copy_to_device(const HostArray<T>& host, DeviceArray<T>& device)
{
    assert(host.size() == device.size());
    detail::copy_to_device_impl(host.data(), device.data(), host.size() * sizeof(T), nullptr);
}

template<typename T>
void copy_to_device_async(const HostArray<T>& host, DeviceArray<T>& device, Stream& s)
{
    assert(host.size() == device.size());
    detail::copy_to_device_impl(host.data(), device.data(), host.size() * sizeof(T), &s);
}

template<typename T>
void copy_to_host(HostArray<T>& host, const DeviceArray<T>& device)
{
    assert(host.size() == device.size());
    detail::copy_to_host_impl(host.data(), device.data(), host.size() * sizeof(T), nullptr);
}

template<typename T>
void copy_to_host_async(HostArray<T>& host, const DeviceArray<T>& device, Stream& s)
{
    assert(host.size() == device.size());
    detail::copy_to_host_impl(host.data(), device.data(), host.size() * sizeof(T), &s);
}

}
