#pragma once

#include <tuple>
#include <type_traits>

#include "transwarp.h"

#include "copy.h"
#include "DeviceArray.h"
#include "HostArray.h"
#include "Stream.h"
#include "RandomGenerator.h"

namespace cutw
{

namespace detail
{

template<bool no_parents>
struct TaskType;

template<>
struct TaskType<true>
{
    static constexpr auto value = transwarp::root;
};

template<>
struct TaskType<false>
{
    static constexpr auto value = transwarp::consume;
};

template<bool no_parents>
struct TaskTypeAny;

template<>
struct TaskTypeAny<true>
{
    static constexpr auto value = transwarp::root;
};

template<>
struct TaskTypeAny<false>
{
    static constexpr auto value = transwarp::consume_any;
};

}

template<typename... Args>
class Output
{
public:
    Output(std::shared_ptr<Args>... args)
        : data_{std::make_tuple(std::move(args)...)}
    {
    }

    const Output& s(std::shared_ptr<Stream> stream)
    {
        stream_ = std::move(stream);
        return *this;
    }

    const std::shared_ptr<Stream>& stream() const
    {
        return stream_;
    }

    const std::tuple<std::shared_ptr<Args>...>& data() const
    {
        return data_;
    }

private:
    std::shared_ptr<Stream> stream_;
    std::tuple<std::shared_ptr<Args>...> data_;
};

template<typename... Args>
Output<Args...> out(std::shared_ptr<Args>... args)
{
    return Output<Args...>{std::move(args)...};
}

template<std::size_t index, typename... Args>
auto get(const Output<Args...>& output) -> decltype(std::get<index>(output.data()))
{
    return std::get<index>(output.data());
}

template<typename... Args>
auto value_task(const std::string& name,
                Output<Args...> output)
    -> decltype(transwarp::make_value_task(std::move(output)))
{
    return transwarp::make_value_task(std::move(output))->named(name);
}

template<typename Functor, typename... Parents>
auto task(const std::string& name,
          Functor&& functor,
          std::shared_ptr<Parents>... parents)
    -> decltype(transwarp::make_task(detail::TaskType<sizeof...(parents) == 0>::value,
                                     std::forward<Functor>(functor),
                                     std::move(parents)...))
{
    return transwarp::make_task(detail::TaskType<sizeof...(parents) == 0>::value,
                                std::forward<Functor>(functor),
                                std::move(parents)...)->named(name);
}

template<typename Functor, typename... Parents>
auto task_any(const std::string& name,
              Functor&& functor,
              std::shared_ptr<Parents>... parents)
    -> decltype(transwarp::make_task(detail::TaskTypeAny<sizeof...(parents) == 0>::value,
                                     std::forward<Functor>(functor),
                                     std::move(parents)...))
{
    return transwarp::make_task(detail::TaskTypeAny<sizeof...(parents) == 0>::value,
                                std::forward<Functor>(functor),
                                std::move(parents)...)->named(name);
}

template<typename T>
class CopyToDevice : public transwarp::functor
{
public:
    Output<DeviceArray<T>>
    operator()(Output<HostArray<T>> host,
               Output<DeviceArray<T>> device) const
    {
        copy_to_device(*get<0>(host), *get<0>(device));
        return device;
    }
};

template<typename T>
class CopyToDeviceAsync : public transwarp::functor
{
public:
    Output<DeviceArray<T>>
    operator()(Output<HostArray<T>> host,
               Output<DeviceArray<T>> device) const
    {
        copy_to_device_async(*get<0>(host), *get<0>(device), *device.stream());
        return device;
    }
};

template<typename T>
class CopyToHost : public transwarp::functor
{
public:
    Output<HostArray<T>>
    operator()(Output<HostArray<T>> host,
               Output<DeviceArray<T>> device) const
    {
        copy_to_host(*get<0>(host), *get<0>(device));
        return host;
    }
};

template<typename T>
class CopyToHostAsync : public transwarp::functor
{
public:
    Output<HostArray<T>>
    operator()(Output<HostArray<T>> host,
               Output<DeviceArray<T>> device) const
    {
        copy_to_host_async(*get<0>(host), *get<0>(device), *host.stream());
        return host;
    }
};

template<typename... Parents>
class SyncStream : public transwarp::functor
{
public:
    Output<Parents...>
    operator()(Output<Parents...> data) const
    {
        data.stream()->sync();
        return data;
    }
};

class CreateRandomGenerator : public transwarp::functor
{
public:
    Output<RandomGenerator>
    operator()(Output<std::size_t> seed) const
    {
        return out(RandomGenerator::create(*get<0>(seed)));
    }
};

template<typename T>
class GenerateRandomUniform : public transwarp::functor
{
public:
    Output<DeviceArray<T>>
    operator()(Output<RandomGenerator> gen, Output<DeviceArray<T>> device) const
    {
        get<0>(gen)->generateUniform(*device.stream(), *get<0>(device));
        return device;
    }
};

}
