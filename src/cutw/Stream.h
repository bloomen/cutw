#pragma once

#include <memory>

struct CUstream_st;

namespace cutw
{

class Stream
{
public:
    Stream();
    ~Stream();

    CUstream_st* get() const;

    void sync();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
