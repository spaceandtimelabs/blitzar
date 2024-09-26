#pragma once

#include "sxt/base/container/span.h"

namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01 
//--------------------------------------------------------------------------------------------------
void sum_polynomial_01(s25t::element& e, basct::cspan<s25t::element> polynomial) noexcept;
} // namespace sxt::prfsk
