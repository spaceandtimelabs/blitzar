#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// async_compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<> async_compute_multiproduct(basct::span<c21t::element_p3> products,
                                          bast::raw_stream_t stream,
                                          basct::cspan<c21t::element_p3> generators,
                                          basct::cspan<unsigned> indexes,
                                          basct::cspan<unsigned> product_sizes) noexcept;
} // namespace sxt::mtxc21
