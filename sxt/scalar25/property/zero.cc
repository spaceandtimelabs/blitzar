#include "sxt/scalar25/property/zero.h"

#include "sxt/base/bit/zero_equality.h"
#include "sxt/scalar25/base/reduce.h"

namespace sxt::s25p {
//--------------------------------------------------------------------------------------------------
// is_zero
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int is_zero(const s25t::element& e) noexcept {
  s25t::element t = e;

  // this `reduce` is necessary for non-reduced values,
  // since `is_zero` do not detect them
  // Ex: 2^252 + 27742317777372353535851937790883648493
  // will not be considered zero by `is_zero`
  s25b::reduce32(t);

  return basbt::is_zero(t.data(), sizeof(t));
}
} // namespace sxt::s25p
