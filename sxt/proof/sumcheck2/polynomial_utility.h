#pragma once

#include <cassert>

#include "sxt/base/container/span.h"
#include "sxt/base/field/element.h"
#include "sxt/base/macro/cuda_callable.h"

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

//--------------------------------------------------------------------------------------------------
// evaluate_polynomial
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
void evaluate_polynomial(T& e, basct::cspan<T> polynomial, const T& x) noexcept {
  if (polynomial.empty()) {
    e = T{};
    return;
  }
  auto i = polynomial.size();
  --i;
  e = polynomial[i];
  while (i > 0) {
    --i;
    muladd(e, e, x, polynomial[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// expand_products
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
CUDA_CALLABLE
void expand_products(basct::span<T> p, const T* mles, unsigned n,
                     unsigned step, basct::cspan<unsigned> terms) noexcept {
  auto num_terms = terms.size();
  assert(
      // clang-format off
      num_terms > 0 && 
      n > step &&
      p.size() == num_terms + 1u
      // clang-format on
  );
  T a, b;
  auto mle_index = terms[0];
  a = *(mles + mle_index * n);
  b = *(mles + mle_index * n + step);
  sub(b, b, a);
  p[0] = a;
  p[1] = b;

  for (unsigned i = 1; i < num_terms; ++i) {
    auto mle_index = terms[i];
    a = *(mles + mle_index * n);
    b = *(mles + mle_index * n + step);
    sub(b, b, a);

    auto c_prev = p[0];
    mul(p[0], c_prev, a);
    for (unsigned pow = 1u; pow < i + 1u; ++pow) {
      auto c = p[pow];
      mul(p[pow], c, a);
      muladd(p[pow], c_prev, b, p[pow]);
      c_prev = c;
    }
    mul(p[i + 1u], c_prev, b);
  }
}
} // namespace sxt::prfsk2
