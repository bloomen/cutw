#include "DeviceArray.h"

#include <cuda.h>

#include "error.h"

namespace cutw
{

namespace detail
{

void device_allocate(void*& data, const std::size_t bytes)
{
    CUTW_CUASSERT(cudaMalloc(&data, bytes));
}

void device_free(void* const data)
{
    CUTW_CUASSERT(cudaFree(data));
}

}

}
