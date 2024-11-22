#include "sxt/proof/sumcheck/fold_gpu.h"

#include "sxt/execution/async/future.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// fold_gpu 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> fold_gpu(basct::span<s25t::element> mles, unsigned n, const s25t::element& r) noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
