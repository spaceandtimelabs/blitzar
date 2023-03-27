#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_scalars
//--------------------------------------------------------------------------------------------------
xena::future<> fold_scalars(basct::span<s25t::element> scalars, const s25t::element& m_low,
                            const s25t::element& m_high, unsigned mid) noexcept;
} // namespace sxt::prfip
