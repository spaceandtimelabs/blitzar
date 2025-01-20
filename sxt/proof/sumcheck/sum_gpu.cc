/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/sum_gpu.h"

#include <cstddef>
#include <iostream>

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/split.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/constant.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/proof/sumcheck/polynomial_mapper2.h"
#include "sxt/proof/sumcheck/reduction_gpu.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
namespace {
template <unsigned Degree> struct polynomial_reducer {
  using value_type = std::array<s25t::element, Degree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i = 0; i < res.size(); ++i) {
      s25o::add(res[i], res[i], e[i]);
    }
  }
};
} // namespace

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel_impl
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize, unsigned NumTerms>
__device__ static void partial_sum_kernel_impl(s25t::element* __restrict__ shared_data,
                                               const s25t::element* __restrict__ mles,
                                               const unsigned* __restrict__ product_terms,
                                               unsigned split, unsigned n) noexcept {
  using Mapper = polynomial_mapper2<NumTerms>;
  using Reducer = polynomial_reducer<NumTerms>;
  using T = Mapper::value_type;
  Mapper mapper{
      .mles = mles,
      .product_terms = product_terms,
      .split = split,
      .n = n,
  };
  auto index = blockIdx.x * (BlockSize * 2) + threadIdx.x;
  auto step = BlockSize * 2 * gridDim.x;
  algr::thread_reduce<Reducer, BlockSize>(reinterpret_cast<T*>(shared_data), mapper, split, step,
                                          threadIdx.x, index);
}

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize>
__global__ static void
partial_sum_kernel(s25t::element* __restrict__ out, const s25t::element* __restrict__ mles,
                   const std::pair<s25t::element, unsigned>* __restrict__ product_table,
                   const unsigned* __restrict__ product_terms, unsigned num_coefficients,
                   unsigned split, unsigned n) noexcept {
  auto product_index = blockIdx.y;
  auto num_terms = product_table[product_index].second;
  auto thread_index = threadIdx.x;
  auto output_step = gridDim.x * gridDim.y;

  // shared data for reduction
  __shared__ s25t::element shared_data[2 * BlockSize * (max_degree_v + 1u)];

  // adjust pointers
  out += blockIdx.x;
  out += gridDim.x * product_index;
  for (unsigned i = 0; i < product_index; ++i) {
    product_terms += product_table[i].second;
  }

  // sum
  basn::constexpr_switch<1, max_degree_v + 1u>(
      num_terms, [&]<unsigned NumTerms>(std::integral_constant<unsigned, NumTerms>) noexcept {
        partial_sum_kernel_impl<BlockSize, NumTerms>(shared_data, mles, product_terms, split, n);
      });

  // write out result
  auto mult = product_table[product_index].first;
  for (unsigned i = thread_index; i < num_coefficients; i += BlockSize) {
    auto output_index = output_step * i;
    if (i < num_terms + 1u) {
      s25o::mul(out[output_index], mult, shared_data[i]);
    } else {
      out[output_index] = s25t::element{};
    }
  }
}

//--------------------------------------------------------------------------------------------------
// partial_sum
//--------------------------------------------------------------------------------------------------
static xena::future<> partial_sum(basct::span<s25t::element> p, basdv::stream& stream,
                                  basct::cspan<s25t::element> mles,
                                  basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                  basct::cspan<unsigned> product_terms, unsigned split,
                                  unsigned n) noexcept {
  auto num_coefficients = p.size();
  auto num_products = product_table.size();
  auto dims = algr::fit_reduction_kernel(split);
  /* emr::async_device_resource resource{stream}; */

  // partials
  memmg::managed_array<s25t::element> partials{num_coefficients * dims.num_blocks * num_products,
                                               memr::get_device_resource()};
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    partial_sum_kernel<BlockSize><<<dim3(dims.num_blocks, num_products, 1), BlockSize, 0, stream>>>(
        partials.data(), mles.data(), product_table.data(), product_terms.data(), num_coefficients,
        split, n);
  });

  // reduce partials
  co_await reduce_sums(p, stream, partials);
}

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       const basit::split_options& options, basct::cspan<s25t::element> mles,
                       unsigned n) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  auto num_mles = mles.size() / n;
  auto num_coefficients = p.size();

  // split
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, mid}, options);

  // sum
  size_t counter = 0;
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](basit::index_range rng) noexcept -> xena::future<> {
        std::println("**************** sum: {}-{}", rng.a(), rng.b());
        basdv::stream stream;
        /* memr::async_device_resource resource{stream}; */

        // copy partial mles to device
        /* memmg::managed_array<s25t::element> partial_mles{&resource}; */
        memmg::managed_array<s25t::element> partial_mles{memr::get_device_resource()};
        copy_partial_mles(partial_mles, stream, mles, n, rng.a(), rng.b());
        auto split = rng.b() - rng.a();
        auto np = partial_mles.size() / num_mles;

        // lookup problem descriptor
        basct::cspan<std::pair<s25t::element, unsigned>> product_table;
        basct::cspan<unsigned> product_terms;
        cache.lookup(product_table, product_terms, stream);

        // compute
        memmg::managed_array<s25t::element> partial_p(num_coefficients);
        co_await partial_sum(partial_p, stream, partial_mles, product_table, product_terms, split,
                             np);

        // fill in the result
        if (counter == 0) {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            p[i] = partial_p[i];
          }
        } else {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            s25o::add(p[i], p[i], partial_p[i]);
          }
        }
        ++counter;
        std::cout << "************************* sum done: " << basdv::get_device() << std::endl;
      });
}

xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       basct::cspan<s25t::element> mles, unsigned n) noexcept {
  basit::split_options options{
      .min_chunk_size = 100'000u,
      .max_chunk_size = 200'000u,
      .split_factor = basdv::get_num_devices(),
  };
  co_await sum_gpu(p, cache, options, mles, n);
}
} // namespace sxt::prfsk
