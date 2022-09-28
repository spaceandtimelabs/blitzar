#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_one_commitments
//--------------------------------------------------------------------------------------------------
void init_precomputed_one_commitments(uint64_t n) noexcept;

//--------------------------------------------------------------------------------------------------
// get_precomputed_one_commitments
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_one_commitments() noexcept;

//--------------------------------------------------------------------------------------------------
// get_precomputed_one_commit
//--------------------------------------------------------------------------------------------------
c21t::element_p3 get_precomputed_one_commit(uint64_t n) noexcept;
} // namespace sxt::sqcgn
