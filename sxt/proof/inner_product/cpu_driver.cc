#include "sxt/proof/inner_product/cpu_driver.h"

#include <algorithm>
#include <memory>
#include <memory_resource>

#include "sxt/base/error/assert.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/ristretto/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/ristretto/precomputed_p3_input_accessor.h"
#include "sxt/proof/inner_product/cpu_workspace.h"
#include "sxt/proof/inner_product/fold.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/verification_computation.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/operation/inner_product.h"
#include "sxt/scalar25/operation/inv.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
static void multiexponentiate(c21t::element_p3& res, basct::cspan<c21t::element_p3> g_vector,
                              basct::cspan<s25t::element> x_vector) noexcept {
  auto n = std::min(g_vector.size(), x_vector.size());
  g_vector = g_vector.subspan(0, n);
  mtxrs::precomputed_p3_input_accessor accessor{g_vector};
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver driver{&accessor, &multiproduct_solver, false};
  memmg::managed_array<c21t::element_p3> inout;
  mtxb::exponent_sequence exponents{
      .element_nbytes = 32,
      .n = n,
      .data = reinterpret_cast<const uint8_t*>(x_vector.data()),
  };
  mtxpi::compute_multiexponentiation(inout, driver,
                                     basct::cspan<mtxb::exponent_sequence>{&exponents, 1});
  res = inout[0];
}

static void multiexponentiate(c21t::element_p3 c_commits[2], const c21t::element_p3& q_value,
                              const s25t::element c_values[2]) noexcept {
  mtxrs::precomputed_p3_input_accessor accessor{{&q_value, 1}};
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver driver{&accessor, &multiproduct_solver, false};
  memmg::managed_array<c21t::element_p3> inout;
  mtxb::exponent_sequence exponents[2] = {
      {
          .element_nbytes = 32,
          .n = 1,
          .data = reinterpret_cast<const uint8_t*>(c_values),
      },
      {
          .element_nbytes = 32,
          .n = 1,
          .data = reinterpret_cast<const uint8_t*>(c_values + 1),
      },
  };
  mtxpi::compute_multiexponentiation(inout, driver, exponents);
  c_commits[0] = inout[0];
  c_commits[1] = inout[1];
}

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
std::unique_ptr<workspace>
cpu_driver::make_workspace(const proof_descriptor& descriptor,
                           basct::cspan<s25t::element> a_vector) const noexcept {
  auto n = a_vector.size();
  auto np_half = descriptor.g_vector.size() / 2;
  SXT_DEBUG_ASSERT(n > 1);

  auto res = std::make_unique<cpu_workspace>();
  res->descriptor = &descriptor;
  res->a_vector0 = a_vector;

  std::pmr::polymorphic_allocator<char> alloc{&res->alloc};
  res->round_index = 0;
  res->g_vector = {
      reinterpret_cast<c21t::element_p3*>(alloc.allocate(np_half * sizeof(c21t::element_p3))),
      np_half,
  };
  auto scalars =
      reinterpret_cast<s25t::element*>(alloc.allocate(2 * np_half * sizeof(s25t::element)));
  res->a_vector = {
      scalars,
      np_half,
  };
  res->b_vector = {
      scalars + np_half,
      np_half,
  };
  return res;
}

//--------------------------------------------------------------------------------------------------
// commit_to_fold
//--------------------------------------------------------------------------------------------------
void cpu_driver::commit_to_fold(rstt::compressed_element& l_value,
                                rstt::compressed_element& r_value, workspace& ws) const noexcept {
  auto& work = static_cast<cpu_workspace&>(ws);
  basct::cspan<c21t::element_p3> g_vector;
  basct::cspan<s25t::element> a_vector;
  basct::cspan<s25t::element> b_vector;
  if (work.round_index == 0) {
    g_vector = work.descriptor->g_vector;
    a_vector = work.a_vector0;
    b_vector = work.descriptor->b_vector;
  } else {
    g_vector = work.g_vector;
    a_vector = work.a_vector;
    b_vector = work.b_vector;
  }
  auto mid = g_vector.size() / 2;
  SXT_DEBUG_ASSERT(mid > 0);

  auto a_low = a_vector.subspan(0, mid);
  auto a_high = a_vector.subspan(mid);
  auto b_low = b_vector.subspan(0, mid);
  auto b_high = b_vector.subspan(mid);
  auto g_low = g_vector.subspan(0, mid);
  auto g_high = g_vector.subspan(mid);

  // c_commits
  s25t::element c_values[2];
  s25o::inner_product(c_values[0], a_low, b_high);
  s25o::inner_product(c_values[1], a_high, b_low);
  c21t::element_p3 c_commits[2];
  multiexponentiate(c_commits, *work.descriptor->q_value, c_values);

  // l_value
  c21t::element_p3 l_value_p;
  multiexponentiate(l_value_p, g_high, a_low);
  c21o::add(l_value_p, l_value_p, c_commits[0]);
  rsto::compress(l_value, l_value_p);

  // r_value
  c21t::element_p3 r_value_p;
  multiexponentiate(r_value_p, g_low, a_high);
  c21o::add(r_value_p, r_value_p, c_commits[1]);
  rsto::compress(r_value, r_value_p);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
void cpu_driver::fold(workspace& ws, const s25t::element& x) const noexcept {
  auto& work = static_cast<cpu_workspace&>(ws);
  basct::cspan<c21t::element_p3> g_vector;
  basct::cspan<s25t::element> a_vector;
  basct::cspan<s25t::element> b_vector;
  if (work.round_index == 0) {
    g_vector = work.descriptor->g_vector;
    a_vector = work.a_vector0;
    b_vector = work.descriptor->b_vector;
  } else {
    g_vector = work.g_vector;
    a_vector = work.a_vector;
    b_vector = work.b_vector;
  }
  auto mid = g_vector.size() / 2;
  SXT_DEBUG_ASSERT(mid > 0);

  ++work.round_index;

  s25t::element x_inv;
  s25o::inv(x_inv, x);

  // a_vector
  fold_scalars(work.a_vector, a_vector, x, x_inv, mid);
  if (mid == 1) {
    // no need to compute the other folded values if we reduce to a single element
    return;
  }

  // b_vector
  fold_scalars(work.b_vector, b_vector, x_inv, x, mid);

  // g_vector
  fold_generators(work.g_vector, g_vector, x_inv, x, mid);
}

//--------------------------------------------------------------------------------------------------
// compute_expected_commitment
//--------------------------------------------------------------------------------------------------
void cpu_driver::compute_expected_commitment(rstt::compressed_element& commit,
                                             const proof_descriptor& descriptor,
                                             basct::cspan<rstt::compressed_element> l_vector,
                                             basct::cspan<rstt::compressed_element> r_vector,
                                             basct::cspan<s25t::element> x_vector,
                                             const s25t::element& ap_value) const noexcept {
  auto num_rounds = l_vector.size();
  auto np = descriptor.g_vector.size();
  // clang-format off
  SXT_DEBUG_ASSERT(
    np > 0 &&
    l_vector.size() == num_rounds &&
    r_vector.size() == num_rounds &&
    x_vector.size() == num_rounds
  );
  // clang-format on
  auto num_exponents = 1 + np + 2 * num_rounds;

  // exponents
  std::vector<s25t::element> exponents(num_exponents);
  compute_verification_exponents(exponents, x_vector, ap_value, descriptor.b_vector);

  // generators
  memmg::managed_array<c21t::element_p3> inout(num_exponents);
  auto iter = inout.data();
  *iter++ = *descriptor.q_value;
  iter = std::copy(descriptor.g_vector.begin(), descriptor.g_vector.end(), iter);
  for (auto& li : l_vector) {
    rsto::decompress(*iter++, li);
  }
  for (auto& ri : r_vector) {
    rsto::decompress(*iter++, ri);
  }

  // commitment
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver driver{nullptr, &multiproduct_solver, false};
  mtxb::exponent_sequence exponent_sequence{
      .element_nbytes = 32,
      .n = num_exponents,
      .data = reinterpret_cast<const uint8_t*>(exponents.data()),
  };
  mtxpi::compute_multiexponentiation(inout, driver, {&exponent_sequence, 1});
  rsto::compress(commit, inout[0]);
}
} // namespace sxt::prfip
