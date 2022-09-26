#pragma once

#include <cstddef>
#include <vector>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_generators
//--------------------------------------------------------------------------------------------------
void init_precomputed_generators(size_t n, bool use_gpu) noexcept;

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_generators() noexcept;

basct::cspan<c21t::element_p3>
get_precomputed_generators(std::vector<c21t::element_p3>& generators_data,
                           size_t length_longest_sequence, size_t offset, bool use_gpu) noexcept;
} // namespace sxt::sqcgn
