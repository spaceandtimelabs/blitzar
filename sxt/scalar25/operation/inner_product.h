#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inner_product
//--------------------------------------------------------------------------------------------------
void inner_product(s25t::element& res, basct::cspan<s25t::element> lhs,
                   basct::cspan<s25t::element> rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// async_inner_product
//--------------------------------------------------------------------------------------------------
xena::future<s25t::element> async_inner_product(basct::cspan<s25t::element> lhs,
                                                basct::cspan<s25t::element> rhs) noexcept;
} // namespace sxt::s25o
