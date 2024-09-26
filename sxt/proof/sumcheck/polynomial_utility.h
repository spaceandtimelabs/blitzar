#pragma once

#include "sxt/base/container/span.h"

namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01 
//--------------------------------------------------------------------------------------------------
void sum_polynomial_01(s25t::element& e, basct::cspan<s25t::element> polynomial) noexcept;

//--------------------------------------------------------------------------------------------------
// evaluate_polynomial
//--------------------------------------------------------------------------------------------------
void evaluate_polynomial(s25t::element& e, basct::cspan<s25t::element> polynomial,
                         const s25t::element& x) noexcept;
} // namespace sxt::prfsk
