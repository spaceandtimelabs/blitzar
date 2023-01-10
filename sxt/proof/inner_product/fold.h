#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_scalars
//--------------------------------------------------------------------------------------------------
void fold_scalars(basct::span<s25t::element>& xp_vector, basct::cspan<s25t::element> x_vector,
                  const s25t::element& m_low, const s25t::element& m_high, size_t mid) noexcept;

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
void fold_generators(basct::span<c21t::element_p3>& gp_vector,
                     basct::cspan<c21t::element_p3> g_vector, const s25t::element& m_low,
                     const s25t::element& m_high, size_t mid) noexcept;
} // namespace sxt::prfip
