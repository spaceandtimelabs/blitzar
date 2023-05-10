/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"
#include "sxt/multiexp/curve21/multiexponentiation.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/generator_fold.h"
#include "sxt/proof/inner_product/generator_fold_kernel.h"
#include "sxt/proof/inner_product/gpu_workspace.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/scalar_fold_kernel.h"
#include "sxt/proof/inner_product/verification_computation_gpu.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/constant/max_bits.h"
#include "sxt/scalar25/operation/inner_product.h"
#include "sxt/scalar25/operation/inv.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// commit_to_fold_partial
//--------------------------------------------------------------------------------------------------
static xena::future<void> commit_to_fold_partial(rstt::compressed_element& commit,
                                                 basct::cspan<c21t::element_p3> g_vector,
                                                 const c21t::element_p3& q_value,
                                                 basct::cspan<s25t::element> u_vector,
                                                 basct::cspan<s25t::element> v_vector) noexcept {
  auto u_commit_fut = mtxc21::async_compute_multiexponentiation(
      g_vector.subspan(0, u_vector.size()), mtxb::to_exponent_sequence(u_vector));
  auto product_fut = s25o::async_inner_product(u_vector, v_vector);
  c21t::element_p3 commit_p;
  c21o::scalar_multiply(commit_p, co_await std::move(product_fut), q_value);
  c21o::add(commit_p, co_await std::move(u_commit_fut), commit_p);
  rsto::compress(commit, commit_p);
}

//--------------------------------------------------------------------------------------------------
// setup_verification_generators
//--------------------------------------------------------------------------------------------------
static xena::future<>
setup_verification_generators(basct::span<c21t::element_p3> generators,
                              const proof_descriptor& descriptor,
                              basct::cspan<rstt::compressed_element> l_vector,
                              basct::cspan<rstt::compressed_element> r_vector) noexcept {
  auto np = descriptor.g_vector.size();
  auto num_rounds = l_vector.size();
  basdv::stream stream;

  // q_value
  basdv::async_copy_host_to_device(basct::subspan(generators, 0, 1),
                                   basct::cspan<c21t::element_p3>{descriptor.q_value, 1}, stream);

  // g_vector
  basdv::async_copy_host_to_device(basct::subspan(generators, 1, np), descriptor.g_vector, stream);

  // l_vector, r_vector
  memmg::managed_array<c21t::element_p3> lr_vector{2 * num_rounds, memr::get_pinned_resource()};
  auto iter = lr_vector.data();
  for (auto& li : l_vector) {
    rsto::decompress(*iter++, li);
  }
  for (auto& ri : r_vector) {
    rsto::decompress(*iter++, ri);
  }
  basdv::async_copy_host_to_device(basct::subspan(generators, np + 1), lr_vector, stream);

  co_await xena::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>>
gpu_driver::make_workspace(const proof_descriptor& descriptor,
                           basct::cspan<s25t::element> a_vector) const noexcept {
  auto res = std::make_unique<gpu_workspace>();
  basdv::stream stream;
  auto alloc = res->a_vector.get_allocator();

  res->descriptor = &descriptor;
  res->round_index = 0;

  // a_vector
  res->a_vector = memmg::managed_array<s25t::element>{
      a_vector.size(),
      alloc,
  };
  basdv::async_copy_host_to_device(res->a_vector, a_vector, stream);

  // b_vector
  res->b_vector = memmg::managed_array<s25t::element>{
      descriptor.b_vector.size(),
      alloc,
  };
  basdv::async_copy_host_to_device(res->b_vector, descriptor.b_vector, stream);

  // g_vector
  res->g_vector = memmg::managed_array<c21t::element_p3>{
      descriptor.g_vector.size(),
      alloc,
  };
  basdv::async_copy_host_to_device(res->g_vector, descriptor.g_vector, stream);

  return xena::await_and_own_stream(std::move(stream), std::unique_ptr<workspace>{std::move(res)});
}

//--------------------------------------------------------------------------------------------------
// commit_to_fold
//--------------------------------------------------------------------------------------------------
xena::future<void> gpu_driver::commit_to_fold(rstt::compressed_element& l_value,
                                              rstt::compressed_element& r_value,
                                              workspace& ws) const noexcept {
  auto& work = static_cast<gpu_workspace&>(ws);
  auto mid = work.g_vector.size() / 2;
  SXT_DEBUG_ASSERT(mid > 0);

  auto a_low = basct::subspan(work.a_vector, 0, mid);
  auto a_high = basct::subspan(work.a_vector, mid);
  auto b_low = basct::subspan(work.b_vector, 0, mid);
  auto b_high = basct::subspan(work.b_vector, mid);
  auto g_low = basct::subspan(work.g_vector, 0, mid);
  auto g_high = basct::subspan(work.g_vector, mid);

  auto l_fut = commit_to_fold_partial(l_value, g_high, *work.descriptor->q_value, a_low, b_high);
  auto r_fut = commit_to_fold_partial(r_value, g_low, *work.descriptor->q_value, a_high, b_low);

  co_await std::move(l_fut);
  co_await std::move(r_fut);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<void> gpu_driver::fold(workspace& ws, const s25t::element& x) const noexcept {
  auto& work = static_cast<gpu_workspace&>(ws);
  auto mid = work.g_vector.size() / 2u;

  ++work.round_index;

  s25t::element x_inv;
  s25o::inv(x_inv, x);

  // a_vector
  auto a_fut = fold_scalars(work.a_vector, x, x_inv, mid);
  work.a_vector.shrink(mid);
  if (mid == 1) {
    // no need to compute the other folded values if we reduce to a single element
    co_await std::move(a_fut);
    co_return;
  }

  // b_vector
  auto b_fut = fold_scalars(work.b_vector, x_inv, x, mid);
  work.b_vector.shrink(mid);

  // g_vector
  unsigned decomposition_data[s25cn::max_bits_v];
  basct::span<unsigned> decomposition{decomposition_data};
  decompose_generator_fold(decomposition, x_inv, x);
  auto g_fut = fold_generators(work.g_vector, decomposition);
  work.g_vector.shrink(mid);

  co_await std::move(a_fut);
  co_await std::move(b_fut);
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
  memmg::managed_array<s25t::element> exponents{num_exponents, memr::get_device_resource()};
  auto fut =
      async_compute_verification_exponents(exponents, x_vector, ap_value, descriptor.b_vector);

  // generators
  memmg::managed_array<c21t::element_p3> generators{num_exponents, memr::get_device_resource()};
  co_await setup_verification_generators(generators, descriptor, l_vector, r_vector);

  // commitment
  co_await std::move(fut);
  auto commit_p = co_await mtxc21::async_compute_multiexponentiation(
      generators, mtxb::to_exponent_sequence(exponents));
  rsto::compress(commit, commit_p);
}
} // namespace sxt::prfip
