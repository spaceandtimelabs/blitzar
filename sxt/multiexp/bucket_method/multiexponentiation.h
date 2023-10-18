#pragma once

#include <algorithm>
#include <limits>
#include <chrono>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/bucket_method/bucket_accumulation.h"
#include "sxt/multiexp/bucket_method/bucket_combination.h"
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
  if (res.empty()) {
    co_return;
  }

  basdv::stream stream;

  // accumulate
  auto t1 = std::chrono::steady_clock::now();
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> bucket_sums{bucket_group_size * num_bucket_groups * num_outputs,
                                      &resource};
  co_await accumulate_buckets<T>(bucket_sums, generators, exponents);
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "accumulation: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0
            << std::endl;

  // reduce buckets
  memmg::managed_array<T> reduced_buckets_dev{bucket_group_size * num_outputs, &resource};
  static unsigned num_threads = 32;
  dim3 block_dims(basn::divide_up(bucket_group_size, num_threads), num_outputs, 1);
  combine_bucket_groups<bucket_group_size, num_bucket_groups>
      <<<block_dims, num_threads, 0, stream>>>(reduced_buckets_dev.data(), bucket_sums.data());
  memmg::managed_array<T> reduced_buckets{reduced_buckets_dev.size(), memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(reduced_buckets, reduced_buckets_dev, stream);
  co_await xendv::await_stream(stream);
  reduced_buckets_dev.reset();

  // combine buckets
  combine_buckets<T>(res, reduced_buckets);
  auto t3 = std::chrono::steady_clock::now();
  std::cout << "rest: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() / 1000.0
            << std::endl;
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate(basct::cspan<Element> generators,
                      basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  memmg::managed_array<Element> res;
  /* co_return res; */
  uint64_t min_n = std::numeric_limits<uint64_t>::max();
  uint64_t max_n = 0;
  for (auto& exponent : exponents) {
    if (exponent.element_nbytes != 32) {
      co_return res;
    }
    min_n = std::min(min_n, exponent.n);
    max_n = std::max(max_n, exponent.n);
  }
  if (min_n != max_n) {
      co_return res;
  }
  auto n = max_n;
  SXT_DEBUG_ASSERT(
      generators.size() >= n
  );
  generators = generators.subspan(0, n);
  memmg::managed_array<const uint8_t*> exponents_p(exponents.size());
  for(size_t output_index=0; output_index<num_outputs; ++output_index) {
    exponents_p[output_index] = exponents[output_index].data;
  }
  res.resize(num_outputs);
  co_await multiexponentiate<Element>(res, generators, exponents_p);
  co_return res;
}
} // namespace sxt::mtxbk
