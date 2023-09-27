#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/bucket_method/accumulate_kernel.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// accumulate_buckets_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> accumulate_buckets_impl(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                       basct::cspan<const uint8_t*> exponents,
                                       basit::index_range rng) noexcept {

  (void)bucket_sums;
  (void)generators;
  (void)exponents;
  (void)rng;
  return {};
}

//--------------------------------------------------------------------------------------------------
// accumulate_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                  basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr size_t bucket_group_size = 255;
  constexpr size_t num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      bucket_sums.size() == bucket_group_size * num_bucket_groups * num_outputs
  );
  for (auto& e : bucket_sums) {
    e = T::identity();
  }
  co_await xendv::concurrent_for_each(
      basit::index_range{0, generators.size()}, [&](const basit::index_range& rng) noexcept {
        return accumulate_buckets_impl(bucket_sums, generators, exponents, rng);
      });
}
} // namespace sxt::mtxbk
