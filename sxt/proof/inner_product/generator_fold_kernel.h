#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept;
} // namespace sxt::prfip
