#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// sum_curve21_elements
//--------------------------------------------------------------------------------------------------
void sum_curve21_elements(basct::span<c21t::element_p3> result,
                          basct::cspan<basct::cspan<uint64_t>> terms,
                          basct::cspan<c21t::element_p3> inputs) noexcept;

//--------------------------------------------------------------------------------------------------
// mul_sum_curve21_elements
//--------------------------------------------------------------------------------------------------
void mul_sum_curve21_elements(basct::span<c21t::element_p3> result,
                              basct::cspan<c21t::element_p3> generators,
                              basct::cspan<mtxb::exponent_sequence> sequences) noexcept;
} // namespace sxt::mtxtst
