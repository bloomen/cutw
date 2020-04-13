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
    auto stream = cutw::Stream::create();
    REQUIRE(stream->get());
}

TEST_CASE("CopyToDevice_CopyToHost")
{
    auto host = cutw::HostArray<float>::create(3);
    host->data()[0] = 1;
    host->data()[1] = 2;
    host->data()[2] = 3;
    auto host_task = cutw::value_task("host", cutw::out(nullptr, host));
    auto dev0_task = cutw::value_task("device", cutw::out(nullptr, cutw::DeviceArray<float>::create(3)));
    auto dev_task = cutw::task("CopyToDevice", cutw::CopyToDevice<float>{}, host_task, dev0_task);
    dev_task->schedule_all();
    auto dev = cutw::get<0>(dev_task->get());
    REQUIRE(dev->data());
    REQUIRE(dev->size() == 3);
    auto host1_task = cutw::value_task("host1", cutw::out(nullptr, cutw::HostArray<float>::create(3)));
    auto host2_task = cutw::task("host2", cutw::CopyToHost<float>{}, host1_task, dev_task);
    host2_task->schedule_all();
    auto host2 = cutw::get<0>(host2_task->get());
    REQUIRE(host2->data());
    REQUIRE(host2->size() == 3);
    REQUIRE(host2->data()[0] == 1);
    REQUIRE(host2->data()[1] == 2);
    REQUIRE(host2->data()[2] == 3);
}

TEST_CASE("CopyToDevice_CopyToHost_async")
{
    auto stream = cutw::Stream::create();
    auto host = cutw::HostArray<float>::create(3);
    host->data()[0] = 1;
    host->data()[1] = 2;
    host->data()[2] = 3;
    auto host_task = cutw::value_task("host", cutw::out(stream, host));
    auto dev0_task = cutw::value_task("device", cutw::out(stream, cutw::DeviceArray<float>::create(3)));
    auto dev_task = cutw::task("CopyToDevice", cutw::CopyToDevice<float>{}, host_task, dev0_task);
    dev_task->schedule_all();
    auto dev = cutw::get<0>(dev_task->get());
    REQUIRE(dev->data());
    REQUIRE(dev->size() == 3);
    auto host1_task = cutw::value_task("host1", cutw::out(stream, cutw::HostArray<float>::create(3)));
    auto host2_task = cutw::task("host2", cutw::CopyToHost<float>{}, host1_task, dev_task);
    auto sync_task = cutw::task("sync", cutw::SyncStream<cutw::HostArray<float>>{}, host2_task);
    sync_task->schedule_all();
    auto host2 = cutw::get<0>(sync_task->get());
    REQUIRE(host2->data());
    REQUIRE(host2->size() == 3);
    REQUIRE(host2->data()[0] == 1);
    REQUIRE(host2->data()[1] == 2);
    REQUIRE(host2->data()[2] == 3);
}

TEST_CASE("CreateRandomGenerator_GenerateRandomUniform")
{
    auto stream = cutw::Stream::create();
    auto dev_task = cutw::value_task("device", cutw::out(stream, cutw::DeviceArray<float>::create(3)));
    auto seed_task = cutw::value_task("seed", cutw::out(nullptr, std::make_shared<std::size_t>(42)));
    auto gen_task = cutw::task("generator", cutw::CreateRandomGenerator{}, seed_task);
    auto uniform_task = cutw::task("uniform", cutw::GenerateRandomUniform<float>{}, gen_task, dev_task);
    auto host1_task = cutw::value_task("host1", cutw::out(stream, cutw::HostArray<float>::create(3)));
    auto host2_task = cutw::task("host2", cutw::CopyToHost<float>{}, host1_task, dev_task);
    auto sync_task = cutw::task("sync", cutw::SyncStream<cutw::HostArray<float>>{}, host2_task);
    sync_task->schedule_all();
    auto host2 = cutw::get<0>(sync_task->get());
    REQUIRE(host2->data());
    REQUIRE(host2->size() == 3);
    REQUIRE(host2->data()[0] > 0.0f);
    REQUIRE(host2->data()[1] > 0.0f);
    REQUIRE(host2->data()[2] > 0.0f);
}
