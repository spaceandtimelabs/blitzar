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
#pragma once

#include <cstddef>

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/split.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/field/element.h"
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
#include "sxt/proof/sumcheck/polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_reducer.h"
#include "sxt/proof/sumcheck/reduction_gpu.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_options
//--------------------------------------------------------------------------------------------------
struct sum_options {
  unsigned min_chunk_size = 100'000u;
  unsigned max_chunk_size = 250'000u;
  unsigned split_factor = unsigned(basdv::get_num_devices());
};

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel_impl
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize, unsigned NumTerms, basfld::element T>
__device__ static void partial_sum_kernel_impl(T* __restrict__ shared_data,
                                               const T* __restrict__ mles,
                                               const unsigned* __restrict__ product_terms,
                                               unsigned split, unsigned n) noexcept {
  using Mapper = polynomial_mapper<NumTerms, T>;
  using Reducer = polynomial_reducer<NumTerms, T>;
  using U = Mapper::value_type;
  Mapper mapper{
      .mles = mles,
      .product_terms = product_terms,
      .split = split,
      .n = n,
  };
  auto index = blockIdx.x * (BlockSize * 2) + threadIdx.x;
  auto step = BlockSize * 2 * gridDim.x;
  algr::thread_reduce<Reducer, BlockSize>(reinterpret_cast<U*>(shared_data), mapper, split, step,
                                          threadIdx.x, index);
}

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize, basfld::element T>
__global__ static void partial_sum_kernel(T* __restrict__ out, const T* __restrict__ mles,
                                          const std::pair<T, unsigned>* __restrict__ product_table,
                                          const unsigned* __restrict__ product_terms,
                                          unsigned num_coefficients, unsigned split,
                                          unsigned n) noexcept {
  auto product_index = blockIdx.y;
  auto num_terms = product_table[product_index].second;
  auto thread_index = threadIdx.x;
  auto output_step = gridDim.x * gridDim.y;

  // shared data for reduction
  __shared__ T shared_data[2 * BlockSize * (max_degree_v + 1u)];

  // adjust pointers
  out += blockIdx.x;
  out += gridDim.x * product_index;
  for (unsigned i = 0; i < product_index; ++i) {
    product_terms += product_table[i].second;
  }

  // sum
  basn::constexpr_switch<1, max_degree_v + 1u>(
      num_terms, [&]<unsigned NumTerms>(std::integral_constant<unsigned, NumTerms>) noexcept {
        partial_sum_kernel_impl<BlockSize, NumTerms, T>(shared_data, mles, product_terms, split, n);
      });

  // write out result
  auto mult = product_table[product_index].first;
  for (unsigned i = thread_index; i < num_coefficients; i += BlockSize) {
    auto output_index = output_step * i;
    if (i < num_terms + 1u) {
      mul(out[output_index], mult, shared_data[i]);
    } else {
      out[output_index] = T{};
    }
  }
}

//--------------------------------------------------------------------------------------------------
// partial_sum
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
static xena::future<> partial_sum(basct::span<T> p, basdv::stream& stream, basct::cspan<T> mles,
                                  basct::cspan<std::pair<T, unsigned>> product_table,
                                  basct::cspan<unsigned> product_terms, unsigned split,
                                  unsigned n) noexcept {
  auto num_coefficients = p.size();
  auto num_products = product_table.size();
  auto dims = algr::fit_reduction_kernel(split);
  memr::async_device_resource resource{stream};

  // partials
  memmg::managed_array<T> partials{num_coefficients * dims.num_blocks * num_products, &resource};
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    partial_sum_kernel<BlockSize><<<dim3(dims.num_blocks, num_products, 1), BlockSize, 0, stream>>>(
        partials.data(), mles.data(), product_table.data(), product_terms.data(), num_coefficients,
        split, n);
  });

  // reduce partials
  co_await reduce_sums<T>(p, stream, partials);
}

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
xena::future<> sum_gpu(basct::span<T> p, device_cache<T>& cache,
                       const basit::split_options& options, basct::cspan<T> mles,
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
        basdv::stream stream;
        memr::async_device_resource resource{stream};

        // copy partial mles to device
        memmg::managed_array<T> partial_mles{&resource};
        copy_partial_mles<T>(partial_mles, stream, mles, n, rng.a(), rng.b());
        auto split = rng.b() - rng.a();
        auto np = partial_mles.size() / num_mles;

        // lookup problem descriptor
        basct::cspan<std::pair<T, unsigned>> product_table;
        basct::cspan<unsigned> product_terms;
        cache.lookup(product_table, product_terms, stream);

        // compute
        memmg::managed_array<T> partial_p(num_coefficients);
        co_await partial_sum<T>(partial_p, stream, partial_mles, product_table, product_terms,
                                split, np);

        // fill in the result
        if (counter == 0) {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            p[i] = partial_p[i];
          }
        } else {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            add(p[i], p[i], partial_p[i]);
          }
        }
        ++counter;
      });
}

template <basfld::element T>
xena::future<> sum_gpu(basct::span<T> p, device_cache<T>& cache, basct::cspan<T> mles,
                       unsigned n) noexcept {
  basit::split_options options{
      .min_chunk_size = 100'000u,
      .max_chunk_size = 200'000u,
      .split_factor = basdv::get_num_devices(),
  };
  co_await sum_gpu<T>(p, cache, options, mles, n);
}

template <basfld::element T>
xena::future<> sum_gpu(basct::span<T> p, basct::cspan<T> mles,
                       basct::cspan<std::pair<T, unsigned>> product_table,
                       basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_host_pointer(p.data()) &&
      basdv::is_active_device_pointer(mles.data()) &&
      basdv::is_active_device_pointer(product_table.data()) &&
      basdv::is_active_device_pointer(product_terms.data())
      // clang-format on
  );
  basdv::stream stream;
  co_await partial_sum<T>(p, stream, mles, product_table, product_terms, mid, n);
}

//--------------------------------------------------------------------------------------------------
// sum_gpu2
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
template <basfld::element T>
xena::future<> sum_gpu2(basct::span<T> p, device_cache<T>& cache,
                        const basit::split_options& options, basct::cspan<T> mles,
                        unsigned n) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  auto num_mles = mles.size() / n;
  auto num_coefficients = p.size();

  // split
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, mid}, options);

  // sum
  size_t counter = 0;
  co_await xendv::for_each_device(
      chunk_first, chunk_last,
      [&](xendv::device_context& ctx, const basit::index_range& rng) noexcept -> xena::future<> {
        std::println(stderr, "sum {}: {}-{}", ctx.device_index, rng.a(), rng.b());
        basdv::stream stream;
        memr::async_device_resource resource{stream};

        // copy partial mles to device
        memmg::managed_array<T> partial_mles{&resource};
        copy_partial_mles<T>(partial_mles, stream, mles, n, rng.a(), rng.b());
        auto split = rng.b() - rng.a();
        auto np = partial_mles.size() / num_mles;

        // lookup problem descriptor
        basct::cspan<std::pair<T, unsigned>> product_table;
        basct::cspan<unsigned> product_terms;
        cache.lookup(product_table, product_terms, stream);

        // compute
        co_await ctx.alt_future.get_future();
        memmg::managed_array<T> partial_p(num_coefficients);
        co_await partial_sum<T>(partial_p, stream, partial_mles, product_table, product_terms,
                                split, np);

        // fill in the result
        if (counter == 0) {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            p[i] = partial_p[i];
          }
        } else {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            add(p[i], p[i], partial_p[i]);
          }
        }
        ++counter;

      });
}
#pragma clang diagnostic pop

template <basfld::element T>
xena::future<> sum_gpu2(basct::span<T> p, device_cache<T>& cache, basct::cspan<T> mles,
                       unsigned n) noexcept {
  auto num_mles = mles.size() / n;
  auto options = basdv::plan_split(num_mles * sizeof(T));
  co_await sum_gpu2<T>(p, cache, options, mles, n);
}
} // namespace sxt::prfsk
