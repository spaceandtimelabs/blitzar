#pragma once

#include <iterator>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/bucket_method/sum.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate_options 
//--------------------------------------------------------------------------------------------------
struct multiexponentiate_options {
  unsigned min_chunk_size = 1'000u;
  unsigned max_chunk_size = 256'000u;
  unsigned bit_width = 8u;
  unsigned split_factor = 1u;
};

//--------------------------------------------------------------------------------------------------
// plan_multiexponentiation 
//--------------------------------------------------------------------------------------------------
void plan_multiexponentiation(multiexponentiate_options& options, unsigned num_outputs,
                              unsigned element_num_bytes, unsigned n) noexcept;

//--------------------------------------------------------------------------------------------------
// multiexponentiate2 
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<>
multiexponentiate2(basct::span<Element> res, const multiexponentiate_options& options,
                   basct::cspan<Element> generators, basct::cspan<const uint8_t*> scalars,
                   unsigned element_num_bytes) noexcept {
  auto num_outputs = res.size();
  auto n = generators.size();

  // sum buckets
  auto rng = basit::index_range{0, n}
                 .min_chunk_size(options.min_chunk_size)
                 .max_chunk_size(options.max_chunk_size);
  auto [chunk_first, chunk_last] = basit::split(rng, options.split_factor);
  auto num_buckets_per_group = (1u << options.bit_width) - 1u;
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, options.bit_width);
  auto num_buckets = num_buckets_per_group * num_bucket_groups * num_outputs;
  auto num_chunks = std::distance(chunk_first, chunk_last);
  memmg::managed_array<Element> bucket_sums_chunks{num_buckets * num_chunks, memr::get_pinned_resource()};
  size_t chunk_index = 0;
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& chunk) noexcept -> xena::future<> {
        auto sums_slice =
            basct::subspan(bucket_sums_chunks, num_buckets * chunk_index, num_buckets);
        memmg::managed_array<const uint8_t*> scalars_slice(num_outputs);
        for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
          scalars_slice[output_index] = scalars[output_index] + element_num_bytes * chunk.a();
        }
        auto generators_slice = generators.subspan(chunk.a(), chunk.size());
        co_await compute_bucket_sums(sums_slice, generators_slice, scalars_slice, element_num_bytes,
                                     options.bit_width);
      });

  // combine chunks
  // reduce bucket sums
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate2
//--------------------------------------------------------------------------------------------------
/**
 * Attempt to compute a multi-exponentiation using the bucket method if the problem dimensions
 * suggest it will give a performance benefit; otherwise, return an empty array.
 */
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate2(basct::cspan<Element> generators,
                      basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  if (num_outputs == 0) {
    co_return {};
  }
  auto n = exponents[0].n;
  if (n == 0) {
    co_return {};
  }
  SXT_RELEASE_ASSERT(generators >= n);
  generators = generators.subspan(0, n);
  memmg::managed_array<const uint8_t*> scalars(num_outputs);
  scalars[0] = exponents[0].data;
  auto element_num_bytes = exponents[0].element_nbytes;
  for (size_t output_index=1; output_index<num_outputs; ++output_index) {
    auto& seq = exponents[output_index];
    if (seq.n != n || seq.element_nbytes != element_num_bytes) {
      co_return {};
    }
    scalars[output_index] = seq.data;
  }
  memmg::managed_array<Element> res{num_outputs, memr::get_pinned_resource()};
  multiexponentiate_options options;
  plan_multiexponentiation(options, num_outputs, element_num_bytes, n);
  co_await multiexponentiate2(res, options, generators, scalars, element_num_bytes);
  co_return res;
}
} // namespace sxt::mtxbk
