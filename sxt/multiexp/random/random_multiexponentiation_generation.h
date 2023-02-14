#pragma once

#include <cstddef>
#include <memory_resource>
#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/memory/alloc.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxrn {
struct random_multiexponentiation_descriptor;
struct random_multiexponentiation_descriptor2;

//--------------------------------------------------------------------------------------------------
// generate_random_multiexponentiation
//--------------------------------------------------------------------------------------------------
void generate_random_multiexponentiation(
    uint64_t& num_inputs, basct::span<mtxb::exponent_sequence> exponents, basm::alloc_t alloc,
    std::mt19937& rng, const random_multiexponentiation_descriptor& descriptor) noexcept;

void generate_random_multiexponentiation(
    uint64_t& num_inputs, basct::span<mtxb::exponent_sequence>& exponents, basm::alloc_t alloc,
    std::mt19937& rng, const random_multiexponentiation_descriptor2& descriptor) noexcept;

void generate_random_multiexponentiation(
    basct::span<uint64_t>& inputs, basct::span<mtxb::exponent_sequence>& exponents,
    basm::alloc_t alloc, std::mt19937& rng,
    const random_multiexponentiation_descriptor2& descriptor) noexcept;
} // namespace sxt::mtxrn
