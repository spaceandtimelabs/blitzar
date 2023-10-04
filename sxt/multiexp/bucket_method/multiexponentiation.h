#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr size_t bucket_group_size = 255;
  constexpr size_t num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      res.size() == num_outputs
  );
  memmg::managed_array<T> bucket_sums{bucket_group_size * num_bucket_groups * num_outputs,
                                      memr::get_pinned_resource()};
  co_await accumulate_buckets(bucket_sums, generators, exponents);
  (void)bucket_sums;

  /* xena::future<> accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators, */
  /*                                   basct::cspan<const uint8_t*> exponents) noexcept { */
  (void)res;
  (void)generators;
  (void)exponents;
}
} // namespace sxt::mtxbk
