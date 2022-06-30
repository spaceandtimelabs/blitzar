#pragma once

#include <random>
#include <cstddef>
#include <memory_resource>

#include "sxt/base/container/span.h"
#include "sxt/base/type/polymorphic_allocator.h"

namespace sxt::mtxb { struct exponent_sequence; }

namespace sxt::mtxrn {
struct random_multiexponentiation_descriptor;

//--------------------------------------------------------------------------------------------------
// generate_random_multiexponentiation
//--------------------------------------------------------------------------------------------------
void generate_random_multiexponentiation(
    uint64_t &num_inputs,
    basct::span<mtxb::exponent_sequence> exponents,
    bast::polymorphic_allocator alloc,
    std::mt19937& rng,
    const random_multiexponentiation_descriptor& descriptor
) noexcept;

}  // namespace sxt::mtxrn
