#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define TRANSWARP_CPP11
#include <cutw/transwarp.h>
namespace tw = transwarp;

#include <cutw/DeviceArray.h>
#include <cutw/HostArray.h>
#include <cutw/Stream.h>
#include <cutw/tasks.h>

TEST_CASE("Stream")
{
    cutw::Stream stream;
    REQUIRE(stream.get());
}

TEST_CASE("HostArray_move_constructor")
{
    cutw::HostArray<float> a{3};
    const auto a_data = a.data();
    auto b = std::move(a);
    REQUIRE(!a.data());
    REQUIRE(a.size() == 0);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("HostArray_move_assignment")
{
    cutw::HostArray<float> a{3};
    const auto a_data = a.data();
    cutw::HostArray<float> b{4};
    b = std::move(a);
    REQUIRE(!a.data());
    REQUIRE(a.size() == 0);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("HostArray_swap")
{
    cutw::HostArray<float> a{3};
    const auto a_data = a.data();
    cutw::HostArray<float> b{4};
    const auto b_data = b.data();
    std::swap(a, b);
    REQUIRE(a.data());
    REQUIRE(a.data() == b_data);
    REQUIRE(a.size() == 4);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("DeviceArray_move_constructor")
{
    cutw::DeviceArray<float> a{3};
    const auto a_data = a.data();
    auto b = std::move(a);
    REQUIRE(!a.data());
    REQUIRE(a.size() == 0);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("DeviceArray_move_assignment")
{
    cutw::DeviceArray<float> a{3};
    const auto a_data = a.data();
    cutw::DeviceArray<float> b{4};
    b = std::move(a);
    REQUIRE(!a.data());
    REQUIRE(a.size() == 0);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("DeviceArray_swap")
{
    cutw::DeviceArray<float> a{3};
    const auto a_data = a.data();
    cutw::DeviceArray<float> b{4};
    const auto b_data = b.data();
    std::swap(a, b);
    REQUIRE(a.data());
    REQUIRE(a.data() == b_data);
    REQUIRE(a.size() == 4);
    REQUIRE(b.data());
    REQUIRE(b.data() == a_data);
    REQUIRE(b.size() == 3);
}

TEST_CASE("CopyToDevice_CopyToHost")
{
    auto host_task = tw::make_value_task(std::make_shared<cutw::HostArray<float>>(3));
    host_task->get()->data()[0] = 1;
    host_task->get()->data()[1] = 2;
    host_task->get()->data()[2] = 3;
    auto dev0_task = tw::make_value_task(std::make_shared<cutw::DeviceArray<float>>(3));
    auto dev_task = tw::make_task(tw::consume, cutw::CopyToDevice<float>{}, host_task, dev0_task);
    dev_task->schedule_all();
    auto dev = dev_task->get();
    REQUIRE(dev->data());
    REQUIRE(dev->size() == 3);
    auto host1_task = tw::make_value_task(std::make_shared<cutw::HostArray<float>>(3));
    auto host2_task = tw::make_task(tw::consume, cutw::CopyToHost<float>{}, host1_task, dev_task);
    host2_task->schedule_all();
    auto host2 = host2_task->get();
    REQUIRE(host2->data());
    REQUIRE(host2->size() == 3);
    REQUIRE(host2->data()[0] == 1);
    REQUIRE(host2->data()[1] == 2);
    REQUIRE(host2->data()[2] == 3);
}

TEST_CASE("CopyToDevice_CopyToHost_async")
{
    auto stream_task = tw::make_value_task(std::make_shared<cutw::Stream>());
    auto host_task = tw::make_value_task(std::make_shared<cutw::HostArray<float>>(3));
    host_task->get()->data()[0] = 1;
    host_task->get()->data()[1] = 2;
    host_task->get()->data()[2] = 3;
    auto dev0_task = tw::make_value_task(std::make_shared<cutw::DeviceArray<float>>(3));
    auto dev_task = tw::make_task(tw::consume, cutw::CopyToDeviceAsync<float>{}, host_task, dev0_task, stream_task);
    dev_task->schedule_all();
    auto dev = dev_task->get();
    REQUIRE(dev->data());
    REQUIRE(dev->size() == 3);
    auto host1_task = tw::make_value_task(std::make_shared<cutw::HostArray<float>>(3));
    auto host2_task = tw::make_task(tw::consume, cutw::CopyToHostAsync<float>{}, host1_task, dev_task, stream_task);
    auto sync_task = tw::make_task(tw::consume, cutw::StreamSync<cutw::HostArray<float>>{}, host2_task, stream_task);
    sync_task->schedule_all();
    auto host2 = sync_task->get();
    REQUIRE(host2->data());
    REQUIRE(host2->size() == 3);
    REQUIRE(host2->data()[0] == 1);
    REQUIRE(host2->data()[1] == 2);
    REQUIRE(host2->data()[2] == 3);
}
