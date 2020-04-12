#include "copy.h"

#include <cuda.h>

#include "error.h"

namespace cutw
{

namespace detail
{

void copy_to_device_impl(const void* const host, void* const device,
                         const std::size_t size, Stream* const s)
{
    if (s)
    {
        CUTW_CUASSERT(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice, s->get()));
    }
    else
    {
        CUTW_CUASSERT(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }
}

void copy_to_host_impl(void* const host, const void* const device,
                       const std::size_t size, Stream* const s)
{
    if (s)
    {
        CUTW_CUASSERT(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, s->get()));
    }
    else
    {
        CUTW_CUASSERT(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }
}

}

}
