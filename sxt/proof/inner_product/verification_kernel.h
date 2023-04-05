#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// compute_g_exponents_partial
//--------------------------------------------------------------------------------------------------
xena::future<> compute_g_exponents_partial(basct::span<s25t::element> g_exponents,
                                           bast::raw_stream_t stream,
                                           basct::cspan<s25t::element> x_sq_vector,
                                           size_t round_first) noexcept;
} // namespace sxt::prfip
