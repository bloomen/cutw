#include "RandomGenerator.h"

#include <curand_kernel.h>

#include "error.h"

namespace cutw
{

struct RandomGenerator::impl
{
    impl()
    {
        CUTW_CURANDASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    }
    ~impl()
    {
        CUTW_CURANDASSERT(curandDestroyGenerator(gen));
    }
    curandGenerator_t gen;
};

RandomGenerator::RandomGenerator(Stream& stream, const std::size_t seed)
    : impl_{new impl}
{
    CUTW_CURANDASSERT(curandSetPseudoRandomGeneratorSeed(impl_->gen, seed));
    CUTW_CURANDASSERT(curandSetStream(impl_->gen, stream.get()));
}

RandomGenerator::~RandomGenerator()
{}

void RandomGenerator::generateUniform(float* const device, const std::size_t n)
{
    CUTW_CURANDASSERT(curandGenerateUniform(impl_->gen, device, n));
}

void RandomGenerator::generateUniform(double* const device, const std::size_t n)
{
    CUTW_CURANDASSERT(curandGenerateUniformDouble(impl_->gen, device, n));
}

curandGenerator_t RandomGenerator::get() const
{
    return impl_->gen;
}

}
