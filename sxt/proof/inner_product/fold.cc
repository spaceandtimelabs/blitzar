#include "sxt/proof/inner_product/fold.h"

#include <cassert>

#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/ristretto/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/ristretto/precomputed_p3_input_accessor.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_scalars
//--------------------------------------------------------------------------------------------------
void fold_scalars(basct::span<s25t::element>& xp_vector, basct::cspan<s25t::element> x_vector,
                  const s25t::element& m_low, const s25t::element& m_high, size_t mid) noexcept {
  assert(x_vector.size() > mid && x_vector.size() <= 2 * mid);
  assert(xp_vector.size() >= mid);
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

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
void fold_generators(basct::span<c21t::element_p3>& gp_vector,
                     basct::cspan<c21t::element_p3> g_vector, const s25t::element& m_low,
                     const s25t::element& m_high, size_t mid) noexcept {
  assert(gp_vector.size() >= mid);
  assert(g_vector.size() == 2 * mid);
  s25t::element m_values[] = {m_low, m_high};
  c21t::element_p3 g_values[2];
  mtxrs::precomputed_p3_input_accessor accessor{g_values};
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver driver{&accessor, &multiproduct_solver, false};
  mtxb::exponent_sequence exponents{
      .element_nbytes = 32,
      .n = 2,
      .data = reinterpret_cast<const uint8_t*>(m_values),
  };
  gp_vector = gp_vector.subspan(0, mid);
  memmg::managed_array<c21t::element_p3> inout;
  for (size_t i = 0; i < mid; ++i) {
    g_values[0] = g_vector[i];
    g_values[1] = g_vector[mid + i];
    mtxpi::compute_multiexponentiation(inout, driver, {&exponents, 1});
    gp_vector[i] = inout[0];
  }
}
} // namespace sxt::prfip
