#pragma once


#define CUTW_CUASSERT(ans) { cutw::cuassert((ans), __FILE__, __LINE__); }

#define CUTW_CURANDASSERT(ans) { cutw::curandassert((ans), __FILE__, __LINE__); }


namespace cutw
{


void cuassert(const int code,
              const char* const file,
              const int line,
              const bool abort = true);


void curandassert(const int code,
                  const char* const file,
                  const int line,
                  const bool abort = true);


}
