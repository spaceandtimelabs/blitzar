#pragma once

#include "sxt/base/container/span.h"

namespace sxt::prft { class transcript; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// verify_sumcheck_no_evaluation 
//--------------------------------------------------------------------------------------------------
bool verify_sumcheck_no_evaluation(s25t::element& expected_sum,
                                   basct::span<s25t::element> evaluation_point,
                                   prft::transcript& transcript, 
                                   basct::span<s25t::element> round_polynomials,
                                   unsigned round_degree) noexcept;
} // namespace sxt::prfsk
