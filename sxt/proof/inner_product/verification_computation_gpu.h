#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// async_compute_verification_exponents
//--------------------------------------------------------------------------------------------------
xena::future<> async_compute_verification_exponents(basct::span<s25t::element> exponents,
                                                    basct::cspan<s25t::element> x_vector,
                                                    const s25t::element& ap_value,
                                                    basct::cspan<s25t::element> b_vector) noexcept;
} // namespace sxt::prfip
