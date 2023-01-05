#include "sxt/scalar25/operation/inner_product.h"

#include <algorithm>

#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inner_product
//--------------------------------------------------------------------------------------------------
void inner_product(s25t::element& res, basct::cspan<s25t::element> lhs,
                   basct::cspan<s25t::element> rhs) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  assert(n > 0);
  s25o::mul(res, lhs[0], rhs[0]);
  for (size_t i = 1; i < n; ++i) {
    s25o::muladd(res, lhs[i], rhs[i], res);
  }
}
} // namespace sxt::s25o
