#pragma once

#include <memory>

struct CUstream_st;

namespace cutw
{

class Stream
{
public:
    static std::shared_ptr<Stream> create()
    {
        return std::shared_ptr<Stream>{new Stream};
    }

    ~Stream();

    CUstream_st* get() const;

    void sync();

private:
    Stream();
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
