#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// decompose_generator_fold
//--------------------------------------------------------------------------------------------------
void decompose_generator_fold(basct::span<unsigned>& res, const s25t::element& m_low,
                              const s25t::element& m_high) noexcept;

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void fold_generators(c21t::element_p3& res, basct::cspan<unsigned> decomposition,
                                   const c21t::element_p3& g_low,
                                   const c21t::element_p3& g_high) noexcept;
} // namespace sxt::prfip
