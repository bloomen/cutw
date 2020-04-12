#include "Stream.h"

#include <cuda.h>

#include "error.h"

namespace cutw
{

struct Stream::impl
{
    impl()
    {
        CUTW_CUASSERT(cudaStreamCreate(&stream));
    }
    ~impl()
    {
        CUTW_CUASSERT(cudaStreamDestroy(stream));
    }
    cudaStream_t stream;
};

Stream::Stream()
    : impl_{new impl}
{}

Stream::~Stream()
{}

cudaStream_t Stream::get() const
{
    return impl_->stream;
}

void Stream::sync()
{
    CUTW_CUASSERT(cudaStreamSynchronize(impl_->stream));
}

}
