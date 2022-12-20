#pragma once

#include "sxt/base/container/span.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::s25t {
class element;
}

namespace sxt::s25rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
void generate_random_element(s25t::element& e, basn::fast_random_number_generator& rng) noexcept;

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<s25t::element> ex,
                              basn::fast_random_number_generator& rng) noexcept;
} // namespace sxt::s25rn
