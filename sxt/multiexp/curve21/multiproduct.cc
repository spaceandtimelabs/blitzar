#include "sxt/multiexp/curve21/multiproduct.h"

#include "sxt/curve21/operation/accumulator.h"
#include "sxt/execution/async/future.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// async_compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<> async_compute_multiproduct(basct::span<c21t::element_p3> products,
                                          bast::raw_stream_t stream,
                                          basct::cspan<c21t::element_p3> generators,
                                          basct::cspan<unsigned> indexes,
                                          basct::cspan<unsigned> product_sizes) noexcept {
  return mtxmpg::compute_multiproduct<c21o::accumulator>(products, stream, generators, indexes,
                                                         product_sizes);
}
} // namespace sxt::mtxc21
