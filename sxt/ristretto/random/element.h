#pragma once

#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::c21t {
struct element_p3;
}
namespace sxt::rstt {
class compressed_element;
}

namespace sxt::rstrn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_element(c21t::element_p3& p, basn::fast_random_number_generator& rng) noexcept;

void generate_random_element(c21t::element_p3& p, std::mt19937& rng) noexcept;

void generate_random_element(rstt::compressed_element& p,
                             basn::fast_random_number_generator& rng) noexcept;

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<c21t::element_p3> px,
                              basn::fast_random_number_generator& rng) noexcept;

void generate_random_elements(basct::span<c21t::element_p3> px, std::mt19937& rng) noexcept;

void generate_random_elements(basct::span<rstt::compressed_element> px,
                              basn::fast_random_number_generator& rng) noexcept;
} // namespace sxt::rstrn
