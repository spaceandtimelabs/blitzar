#pragma once

#include <cstddef>

#include "sxt/base/field/element.h"
#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
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
#include "sxt/proof/sumcheck2/polynomial_reducer.h"
#include "sxt/proof/sumcheck/constant.h"
/* #include "sxt/proof/sumcheck/device_cache.h" */
/* #include "sxt/proof/sumcheck/mle_utility.h" */
/* #include "sxt/proof/sumcheck/polynomial_mapper.h" */
/* #include "sxt/proof/sumcheck/reduction_gpu.h" */

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// sum_options
//--------------------------------------------------------------------------------------------------
struct sum_options {
  unsigned min_chunk_size = 100'000u;
  unsigned max_chunk_size = 250'000u;
  unsigned split_factor = unsigned(basdv::get_num_devices());
};

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
#if 0
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
        basdv::stream stream;
        memr::async_device_resource resource{stream};

        // copy partial mles to device
        memmg::managed_array<s25t::element> partial_mles{&resource};
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
#endif

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
  /* co_await partial_sum(p, stream, mles, product_table, product_terms, mid, n); */
  return {};
}
} // namespace sxt::prfsk2
