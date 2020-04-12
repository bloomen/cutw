#include "error.h"

#include <cuda.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>
#include <string>

namespace
{

const char* curandGetErrorString(const curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

}

namespace cutw
{

void
cuassert(const int code,
              const char* const file,
              const int line,
              const bool abort)
{
   if (code != cudaSuccess)
   {
      std::string msg = "cutw: cuassert: ";
      msg += cudaGetErrorString(static_cast<cudaError_t>(code));
      msg += " @ ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      std::cerr << msg << std::endl;
      if (abort)
      {
          assert(false);
          std::terminate();
      }
   }
}

void
curandassert(const int code,
             const char* const file,
             const int line,
             const bool abort)
{
   if (code != CURAND_STATUS_SUCCESS)
   {
      std::string msg = "cutw: curandassert: ";
      msg += curandGetErrorString(static_cast<curandStatus_t>(code));
      msg += " @ ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      std::cerr << msg << std::endl;
      if (abort)
      {
          assert(false);
          std::terminate();
      }
   }
}

}
