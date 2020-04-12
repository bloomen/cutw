#pragma once

#include "transwarp.h"

#include "copy.h"
#include "DeviceArray.h"
#include "HostArray.h"
#include "Stream.h"

namespace cutw
{

template<typename T>
class CopyToDevice : public transwarp::functor
{
public:
    std::shared_ptr<cutw::DeviceArray<T>>
    operator()(std::shared_ptr<cutw::HostArray<T>> host,
               std::shared_ptr<cutw::DeviceArray<T>> device) const
    {
        cutw::copy_to_device(*host, *device);
        return device;
    }
};

template<typename T>
class CopyToDeviceAsync : public transwarp::functor
{
public:
    std::shared_ptr<cutw::DeviceArray<T>>
    operator()(std::shared_ptr<cutw::HostArray<T>> host,
               std::shared_ptr<cutw::DeviceArray<T>> device,
               std::shared_ptr<cutw::Stream> stream) const
    {
        cutw::copy_to_device_async(*host, *device, *stream);
        return device;
    }
};

template<typename T>
class CopyToHost : public transwarp::functor
{
public:
    std::shared_ptr<cutw::HostArray<T>>
    operator()(std::shared_ptr<cutw::HostArray<T>> host,
               std::shared_ptr<cutw::DeviceArray<T>> device) const
    {
        cutw::copy_to_host(*host, *device);
        return host;
    }
};

template<typename T>
class CopyToHostAsync : public transwarp::functor
{
public:
    std::shared_ptr<cutw::HostArray<T>>
    operator()(std::shared_ptr<cutw::HostArray<T>> host,
               std::shared_ptr<cutw::DeviceArray<T>> device,
               std::shared_ptr<cutw::Stream> stream) const
    {
        cutw::copy_to_host_async(*host, *device, *stream);
        return host;
    }
};

template<typename Data>
class StreamSync : public transwarp::functor
{
public:
    std::shared_ptr<Data>
    operator()(std::shared_ptr<Data> data,
               std::shared_ptr<cutw::Stream> stream) const
    {
        stream->sync();
        return data;
    }
};

}
