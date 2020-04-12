#include "HostArray.h"

#include <cuda.h>

#include "error.h"

namespace cutw
{

namespace detail
{

void host_allocate(void*& data, const std::size_t bytes)
{
    CUTW_CUASSERT(cudaMallocHost(&data, bytes));
}

void host_free(void* const data)
{
    CUTW_CUASSERT(cudaFreeHost(data));
}

}

}
