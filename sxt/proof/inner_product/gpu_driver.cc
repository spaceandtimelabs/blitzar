/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/proof/inner_product/gpu_driver.h"

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"
#include "sxt/multiexp/curve/multiexponentiation.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/generator_fold.h"
#include "sxt/proof/inner_product/generator_fold_kernel.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/scalar_fold_kernel.h"
#include "sxt/proof/inner_product/verification_computation_gpu.h"
#include "sxt/proof/inner_product/workspace.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/constant/max_bits.h"
#include "sxt/scalar25/operation/inner_product.h"
#include "sxt/scalar25/operation/inv.h"

namespace sxt::prfip {
static int* f() {
  int res = 123;
  return &res;
}

//--------------------------------------------------------------------------------------------------
// commit_to_fold_partial
//--------------------------------------------------------------------------------------------------
static xena::future<void> commit_to_fold_partial(rstt::compressed_element& commit,
                                                 basct::cspan<c21t::element_p3> g_vector,
                                                 const c21t::element_p3& q_value,
                                                 basct::cspan<s25t::element> u_vector,
                                                 basct::cspan<s25t::element> v_vector) noexcept {
  std::cout << *f() << std::endl;
  auto u_exponents = mtxb::to_exponent_sequence(u_vector);
  auto u_commit_fut = mtxcrv::async_compute_multiexponentiation<c21t::element_p3>(
      g_vector.subspan(0, u_vector.size()), u_exponents);
  auto product_fut = s25o::async_inner_product(u_vector, v_vector);
  c21t::element_p3 commit_p;
  c21o::scalar_multiply(commit_p, co_await std::move(product_fut), q_value);
  c21o::add(commit_p, co_await std::move(u_commit_fut), commit_p);
  rsto::compress(commit, commit_p);
}

//--------------------------------------------------------------------------------------------------
// setup_verification_generators
//--------------------------------------------------------------------------------------------------
static void
setup_verification_generators(basct::span<c21t::element_p3> generators,
                              const proof_descriptor& descriptor,
                              basct::cspan<rstt::compressed_element> l_vector,
                              basct::cspan<rstt::compressed_element> r_vector) noexcept {
  // q_value
  generators[0] = *descriptor.q_value;

  // g_vector
  auto iter =
      std::copy(descriptor.g_vector.begin(), descriptor.g_vector.end(), generators.begin() + 1);

  // l_vector, r_vector
  for (auto& li : l_vector) {
    rsto::decompress(*iter++, li);
  }
  for (auto& ri : r_vector) {
    rsto::decompress(*iter++, ri);
  }
}

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
std::unique_ptr<workspace>
gpu_driver::make_workspace(const proof_descriptor& descriptor,
                           basct::cspan<s25t::element> a_vector) const noexcept {
  auto res = std::make_unique<workspace>(memr::get_pinned_resource());
  res->descriptor = &descriptor;
  res->a_vector0 = a_vector;
  init_workspace(*res);
  return res;
}

//--------------------------------------------------------------------------------------------------
// commit_to_fold
//--------------------------------------------------------------------------------------------------
xena::future<void> gpu_driver::commit_to_fold(rstt::compressed_element& l_value,
                                              rstt::compressed_element& r_value,
                                              workspace& ws) const noexcept {
  auto& work = static_cast<workspace&>(ws);
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

  auto l_fut = commit_to_fold_partial(l_value, g_high, *work.descriptor->q_value, a_low, b_high);
  co_await commit_to_fold_partial(r_value, g_low, *work.descriptor->q_value, a_high, b_low);

  co_await std::move(l_fut);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<void> gpu_driver::fold(workspace& ws, const s25t::element& x) const noexcept {
  auto& work = static_cast<workspace&>(ws);
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

  // g_vector
  unsigned decomposition_data[s25cn::max_bits_v];
  basct::span<unsigned> decomposition{decomposition_data};
  decompose_generator_fold(decomposition, x_inv, x);
  work.g_vector = work.g_vector.subspan(0, mid);
  auto g_fut = async_fold_generators(work.g_vector, g_vector, decomposition);

  // a_vector
  work.a_vector = work.a_vector.subspan(0, mid);
  auto a_fut = async_fold_scalars(work.a_vector, a_vector, x, x_inv);

  // b_vector
  work.b_vector = work.b_vector.subspan(0, mid);
  co_await async_fold_scalars(work.b_vector, b_vector, x_inv, x);

  co_await std::move(a_fut);
  co_await std::move(g_fut);
}

//--------------------------------------------------------------------------------------------------
// compute_expected_commitment
//--------------------------------------------------------------------------------------------------
xena::future<void> gpu_driver::compute_expected_commitment(
    rstt::compressed_element& commit, const proof_descriptor& descriptor,
    basct::cspan<rstt::compressed_element> l_vector,
    basct::cspan<rstt::compressed_element> r_vector, basct::cspan<s25t::element> x_vector,
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
  if (num_rounds < 6) {
    cpu_driver drv;
    co_return co_await drv.compute_expected_commitment(commit, descriptor, l_vector, r_vector,
                                                       x_vector, ap_value);
  }
  auto num_exponents = 1 + np + 2 * num_rounds;

  // exponents
  memmg::managed_array<s25t::element> exponents(num_exponents, memr::get_pinned_resource());
  auto fut =
      async_compute_verification_exponents(exponents, x_vector, ap_value, descriptor.b_vector);

  // generators
  memmg::managed_array<c21t::element_p3> generators(num_exponents, memr::get_pinned_resource());
  setup_verification_generators(generators, descriptor, l_vector, r_vector);

  // commitment
  co_await std::move(fut);
  auto exponent_sequence = mtxb::to_exponent_sequence(exponents);
  auto commit_p = co_await mtxcrv::async_compute_multiexponentiation<c21t::element_p3>(
      generators, exponent_sequence);
  rsto::compress(commit, commit_p);
}
} // namespace sxt::prfip
