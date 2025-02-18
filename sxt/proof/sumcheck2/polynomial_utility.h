#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/field/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01
//--------------------------------------------------------------------------------------------------
// Given a polynomial
//    f_a(X) = a[0] + a[1] * X + a[2] * X^2 + ...
// compute the sum
//    f_a(0) + f_a(1)
template <basfld::element T>
void sum_polynomial_01(T& e, basct::cspan<T> polynomial) noexcept {
  if (polynomial.empty()) {
    e = T{};
    return;
  }
  e = polynomial[0];
  for (unsigned i = 0; i < polynomial.size(); ++i) {
    add(e, e, polynomial[i]);
  }
}
} // namespace sxt::prfsk2
