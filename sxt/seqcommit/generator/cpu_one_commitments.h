#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// cpu_get_one_commitments
//--------------------------------------------------------------------------------------------------
void cpu_get_one_commitments(basct::span<c21t::element_p3> one_commitments) noexcept;

//--------------------------------------------------------------------------------------------------
// cpu_get_one_commit
//--------------------------------------------------------------------------------------------------
c21t::element_p3 cpu_get_one_commit(c21t::element_p3 prev_commit, uint64_t n,
                                    uint64_t offset) noexcept;

c21t::element_p3 cpu_get_one_commit(uint64_t n) noexcept;
} // namespace sxt::sqcgn
