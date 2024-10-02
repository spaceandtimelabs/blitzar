#include "sxt/proof/sumcheck/polynomial_utility.h"

#include <cassert>

#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01 
//--------------------------------------------------------------------------------------------------
void sum_polynomial_01(s25t::element& e, basct::cspan<s25t::element> polynomial) noexcept {
  if (polynomial.empty()) {
    e = s25t::element{};
  }
  e = polynomial[0];
  for (unsigned i=1; i<polynomial.size(); ++i) {
    s25o::add(e, e, polynomial[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// evaluate_polynomial
//--------------------------------------------------------------------------------------------------
void evaluate_polynomial(s25t::element& e, basct::cspan<s25t::element> polynomial,
                         const s25t::element& x) noexcept {
  if (polynomial.empty()) {
    e = s25t::element{};
  }
  auto i = polynomial.size();
  --i;
  e = polynomial[i];
  while (i > 0) {
    --i;
    s25o::muladd(e, e, x, polynomial[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// expand_products 
//--------------------------------------------------------------------------------------------------
void expand_products(basct::span<s25t::element> p, const s25t::element* mles, unsigned n,
                     unsigned step, basct::cspan<unsigned> terms) noexcept {
  auto num_terms = terms.size();
  assert(num_terms > 0 && p.size() == num_terms + 1u);
  s25t::element a, b;
  auto index = terms[0];
  a = *(mles + index * n);
  b = *(mles + index * n + step);
  s25o::sub(b, b, a);
  p[0] = a;
  p[1] = b;

  for (unsigned i=1; i<num_terms; ++i) {
    auto index = terms[i];
    a = *(mles + index * n);
    b = *(mles + index * n + step);
    s25o::sub(b, b, a);

    auto c_prev = p[0];
    s25o::mul(p[0], c_prev, a);
    for (unsigned pow = 1u; pow < i + 1u; ++pow) {
       auto c = p[pow];
       s25o::mul(p[pow], c, a);
       s25o::muladd(p[pow], c_prev, b, p[pow]); 
       c_prev = c;
    }
    s25o::mul(p[i+1u], c_prev, b);
  }
}
} // namespace sxt::prfsk
