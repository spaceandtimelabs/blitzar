#pragma once

#include <iterator>
#include <chrono>
#include <print>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/field51/type/literal.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/bucket_method/reduction.h"
#include "sxt/multiexp/bucket_method/sum3.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate_options 
//--------------------------------------------------------------------------------------------------
struct multiexponentiate_options3 {
  unsigned min_chunk_size = 1'000u;
  unsigned max_chunk_size = 256'000u;
  unsigned bit_width = 4u;
  unsigned split_factor = 1u;
};

//--------------------------------------------------------------------------------------------------
// bucket_sum_combination_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void bucket_sum_combination_kernel3(T* __restrict__ sums,
                                                  const T* __restrict__ chunk_sums,
                                                  unsigned num_buckets, unsigned num_chunks,
                                                  unsigned bucket_index) noexcept {
  T sum = chunk_sums[bucket_index];
  for (unsigned chunk_index = 1; chunk_index < num_chunks; ++chunk_index) {
    auto e = chunk_sums[num_buckets * chunk_index + bucket_index];
    add_inplace(sum, e);
  }
  sums[bucket_index] = sum;
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<>
multiexponentiate3(basct::span<Element> res, const multiexponentiate_options3& options,
                   basct::cspan<Element> generators, basct::cspan<const uint8_t*> scalars,
                   unsigned element_num_bytes) noexcept {
  std::print("********\n");
  auto num_outputs = res.size();
  auto n = generators.size();

  // sum buckets
  auto t1 = std::chrono::steady_clock::now();
  auto rng = basit::index_range{0, n}
                 .min_chunk_size(options.min_chunk_size)
                 .max_chunk_size(options.max_chunk_size);
  auto [chunk_first, chunk_last] = basit::split(rng, options.split_factor);
  auto num_buckets_per_group = (1u << options.bit_width) - 1u;
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, options.bit_width);
  auto num_buckets = num_buckets_per_group * num_bucket_groups * num_outputs;
  auto num_chunks = std::distance(chunk_first, chunk_last);
  memmg::managed_array<Element> bucket_sums_chunks{num_buckets * num_chunks,
                                                   memr::get_pinned_resource()};
  size_t chunk_index = 0;
  auto t2 = std::chrono::steady_clock::now();
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& chunk) noexcept -> xena::future<> {
        auto sums_slice =
            basct::subspan(bucket_sums_chunks, num_buckets * chunk_index, num_buckets);
        memmg::managed_array<const uint8_t*> scalars_slice(num_outputs);
        for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
          scalars_slice[output_index] = scalars[output_index] + element_num_bytes * chunk.a();
        }
        auto generators_slice = generators.subspan(chunk.a(), chunk.size());
        ++chunk_index;
        co_await compute_bucket_sums3(sums_slice, generators_slice, scalars_slice,
                                      element_num_bytes, options.bit_width);
      });

  auto t3 = std::chrono::steady_clock::now();
  // combine chunks
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<Element> bucket_sums{num_buckets, &resource};
  memmg::managed_array<Element> bucket_sums_chunks_dev{bucket_sums_chunks.size(), &resource};
  basdv::async_copy_host_to_device(bucket_sums_chunks_dev, bucket_sums_chunks, stream);
  auto f =
      [
          // clang-format off
    sums = bucket_sums.data(), 
    chunk_sums = bucket_sums_chunks_dev.data(),
    num_chunks = num_chunks
          // clang-format on
  ] __device__
      __host__(unsigned num_buckets, unsigned bucket_index) noexcept {
        bucket_sum_combination_kernel3(sums, chunk_sums, num_buckets, num_chunks, bucket_index);
      };
  algi::launch_for_each_kernel(stream, f, num_buckets);
  bucket_sums_chunks_dev.reset();

  // reduce bucket sums
  reduce_buckets<Element>(res, stream, bucket_sums, options.bit_width);
  co_await xendv::await_stream(std::move(stream));
  auto t4 = std::chrono::steady_clock::now();
  auto mp = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count() / 1000.0;
  auto rest = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() / 1000.0;
  auto total = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() / 1000.0;
  std::print("total = {}\n", total);
  std::print("mp = {}\n", mp);
  std::print("rest = {}\n", rest);
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate3
//--------------------------------------------------------------------------------------------------
/**
 * Attempt to compute a multi-exponentiation using the bucket method if the problem dimensions
 * suggest it will give a performance benefit; otherwise, return an empty array.
 */
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate3(basct::cspan<Element> generators,
                      basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  memmg::managed_array<Element> res{memr::get_pinned_resource()};
  if (num_outputs == 0) {
    co_return res;
  }
  auto n = exponents[0].n;
  if (n == 0 || exponents[0].is_signed) {
    co_return res;
  }
  SXT_RELEASE_ASSERT(generators.size() >= n);
  generators = generators.subspan(0, n);
  memmg::managed_array<const uint8_t*> scalars(num_outputs);
  scalars[0] = exponents[0].data;
  auto element_num_bytes = exponents[0].element_nbytes;
  if (element_num_bytes != 32u) {
    co_return res;
  }
  for (size_t output_index=1; output_index<num_outputs; ++output_index) {
    auto& seq = exponents[output_index];
    if (seq.n != n || seq.element_nbytes != element_num_bytes || seq.is_signed) {
      co_return res;
    }
    scalars[output_index] = seq.data;
  }
  res.resize(num_outputs);
  multiexponentiate_options3 options;
  (void)options;
  /* co_await multiexponentiate3<Element>(res, options, generators, scalars, element_num_bytes); */
  co_return res;
}
} // namespace sxt::mtxbk
