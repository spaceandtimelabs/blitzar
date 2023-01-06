#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/memory/alloc.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// generate_random_product
//--------------------------------------------------------------------------------------------------
void generate_random_product(proof_descriptor& descriptor, basct::cspan<s25t::element>& a_vector,
                             basn::fast_random_number_generator& rng, basm::alloc_t alloc,
                             size_t n) noexcept;
} // namespace sxt::prfip
