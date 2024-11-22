#pragma once

#include "sxt/execution/async/future_fwd.h"
#include "sxt/base/container/span.h"

namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// fold_gpu 
//--------------------------------------------------------------------------------------------------
xena::future<> fold_gpu(basct::span<s25t::element> mles, unsigned n, const s25t::element& r) noexcept;
} // namespace sxt::prfsk
