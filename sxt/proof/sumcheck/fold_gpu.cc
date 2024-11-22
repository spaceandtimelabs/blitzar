#include "sxt/proof/sumcheck/fold_gpu.h"

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/scalar25/operation/sub.h"
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
  using s25t::operator""_s25;
  auto num_mles = mles.size() / n;
  SXT_DEBUG_ASSERT(
      mles.size() == num_mles * n
  );
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
