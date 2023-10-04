#pragma once

#include "sxt/base/num/divide_up.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/bucket_method/bucket_accumulation.h"
#include "sxt/multiexp/bucket_method/combination_kernel.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr unsigned bucket_group_size = 255;
  constexpr unsigned num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      res.size() == num_outputs
  );
  basdv::stream stream;

  // accumulate
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> bucket_sums{bucket_group_size * num_bucket_groups * num_outputs,
                                      &resource};
  co_await accumulate_buckets(bucket_sums, generators, exponents);

  // reduce buckets
  memmg::managed_array<T> reduced_buckets_dev{bucket_group_size * num_outputs, &resource};
  static unsigned num_threads = 32;
  dim3 block_dims(basn::divide_up(bucket_group_size, num_threads), num_outputs, 1);
  combine_bucket_groups<bucket_group_size, num_bucket_groups>
      <<<block_dims, num_threads, 0, stream>>>(reduced_buckets_dev.data(), bucket_sums.data());

  (void)reduced_buckets_dev;
  (void)bucket_sums;

  (void)res;
  (void)generators;
  (void)exponents;
}
} // namespace sxt::mtxbk
