#pragma once

#include <random>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::f51t {
class element;
}

namespace sxt::f51rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_element(f51t::element& e, basn::fast_random_number_generator& rng) noexcept;

void generate_random_element(f51t::element& e, std::mt19937& rng) noexcept;
} // namespace sxt::f51rn
