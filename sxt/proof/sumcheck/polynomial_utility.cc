#include "sxt/proof/sumcheck/polynomial_utility.h"

#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/operation/add.h"
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
} // namespace sxt::prfsk
