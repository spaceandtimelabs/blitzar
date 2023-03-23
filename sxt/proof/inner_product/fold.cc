#include "sxt/proof/inner_product/fold.h"

#include "sxt/base/error/assert.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_scalars
//--------------------------------------------------------------------------------------------------
void fold_scalars(basct::span<s25t::element>& xp_vector, basct::cspan<s25t::element> x_vector,
                  const s25t::element& m_low, const s25t::element& m_high, size_t mid) noexcept {
  SXT_DEBUG_ASSERT(x_vector.size() > mid && x_vector.size() <= 2 * mid);
  SXT_DEBUG_ASSERT(xp_vector.size() >= mid);
  xp_vector = xp_vector.subspan(0, mid);
  auto p = x_vector.size() - mid;
  for (size_t i = 0; i < p; ++i) {
    auto& xp_i = xp_vector[i];
    s25o::mul(xp_i, m_low, x_vector[i]);
    s25o::muladd(xp_i, m_high, x_vector[mid + i], xp_i);
  }
  // If x_vector is not a power of 2, then we perform the fold as if x_vector were padded
  // with zeros until it was a power of 2. Here, we do the operations for the padded elements
  // of the fold (if any).
  for (size_t i = p; i < mid; ++i) {
    s25o::mul(xp_vector[i], m_low, x_vector[i]);
  }
}
} // namespace sxt::prfip
