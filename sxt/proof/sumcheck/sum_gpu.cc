#include "sxt/proof/sumcheck/sum_gpu.h"

#include "sxt/execution/async/future.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_gpu 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> sum_gpu(basct::span<s25t::element> p, basct::cspan<s25t::element> mles,
                       basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                       basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
