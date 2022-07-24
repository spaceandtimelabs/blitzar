#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_generators
//--------------------------------------------------------------------------------------------------
void init_precomputed_generators(size_t n) noexcept;

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_generators() noexcept;
} // namespace sxt::sqcgn
