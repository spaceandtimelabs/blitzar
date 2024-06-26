/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include <concepts>
#include <iterator>
#include <numeric>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/log/log.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/combination.h"
#include "sxt/multiexp/pippenger2/partition_product.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"
#include "sxt/multiexp/pippenger2/reduce.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate_options
//--------------------------------------------------------------------------------------------------
struct multiexponentiate_options {
  unsigned split_factor = 1;
  unsigned min_chunk_size = 64;
  unsigned max_chunk_size = 1024;
};

//--------------------------------------------------------------------------------------------------
// multiexponentiate_no_chunks
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
multiexponentiate_no_chunks(basct::span<T> res, const partition_table_accessor<U>& accessor,
                            unsigned element_num_bytes, basct::cspan<uint8_t> scalars) noexcept {
  auto num_outputs = res.size();
  auto n = scalars.size() / (num_outputs * element_num_bytes);
  auto num_products = num_outputs * element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % (num_outputs * element_num_bytes) == 0
      // clang-format on
  );

  // compute bitwise products
  basl::info("computing {} bitwise multiexponentiation products of length {}", num_products, n);
  memmg::managed_array<T> products(num_products, memr::get_device_resource());
  co_await async_partition_product<T>(products, accessor, scalars, 0);

  // reduce products
  basl::info("reducing {} products to {} outputs", num_products, num_products);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  reduce_products<T>(res_dev, stream, products);
  products.reset();

  // copy result
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
  basl::info("completed {} reductions", num_outputs);
}

template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
multiexponentiate_no_chunks(basct::span<T> res, const partition_table_accessor<U>& accessor,
                            basct::cspan<unsigned> output_bit_table, unsigned num_products,
                            basct::cspan<uint8_t> scalars) noexcept {
  auto num_outputs = res.size();
  auto num_bytes_per_output = basn::divide_up<size_t>(num_products, 8);
  auto n = scalars.size() / (num_outputs * num_bytes_per_output);
#if 0
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % (num_outputs * element_num_bytes) == 0
      // clang-format on
  );
#endif

  // compute bitwise products
  basl::info("computing {} bitwise multiexponentiation products of length {}", num_products, n);
  memmg::managed_array<T> products(num_products, memr::get_device_resource());
  co_await async_partition_product<T>(products, accessor, scalars, 0);

  // reduce products
  basl::info("reducing {} products to {} outputs", num_products, num_products);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  reduce_products<T>(res_dev, stream, output_bit_table, products);
  products.reset();

  // copy result
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate_product_step 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
multiexponentiate_product_step(basct::span<T> products, basdv::stream& reduction_stream,
                               const partition_table_accessor<U>& accessor, unsigned num_products,
                               unsigned num_output_bytes, basct::cspan<uint8_t> scalars,
                               const multiexponentiate_options& options) noexcept {
  auto n = scalars.size() / num_output_bytes;

  // compute bitwise products
  //
  // We split the work by groups of generators so that a single chunk will process
  // all the outputs for those generators. This minimizes the amount of host->device
  // copying we need to do for the table of precomputed sums.
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}
                                                    .chunk_multiple(16)
                                                    .min_chunk_size(options.min_chunk_size)
                                                    .max_chunk_size(options.max_chunk_size),
                                                options.split_factor);
  auto num_chunks = std::distance(chunk_first, chunk_last);
  basl::info("computing {} bitwise multiexponentiation products of length {} using {} chunks",
             num_products, n, num_chunks);

  // handle no chunk case
  if (num_chunks == 1) {
    co_await async_partition_product<T>(products, accessor, scalars, 0);
    co_return;
  }

  // handle multiple chunks
  memmg::managed_array<T> partial_products{num_products * num_chunks, memr::get_pinned_resource()};
  size_t chunk_index = 0;
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        basl::info("computing {} multiproducts for generators [{}, {}] on device {}", num_products,
                   rng.a(), rng.b(), basdv::get_device());
        memmg::managed_array<T> partial_products_dev{num_products, memr::get_device_resource()};
        auto scalars_slice =
            scalars.subspan(num_output_bytes * rng.a(), rng.size() * num_output_bytes);
        co_await async_partition_product<T>(partial_products_dev, accessor, scalars_slice, rng.a());
        basdv::stream stream;
        basdv::async_copy_device_to_host(
            basct::subspan(partial_products, num_products * chunk_index, num_products),
            partial_products_dev, stream);
        ++chunk_index;
        co_await xendv::await_stream(stream);
      });

  // combine the partial products
  memr::async_device_resource resource{reduction_stream};
  memmg::managed_array<T> partial_products_dev{partial_products.size(), &resource};
  basdv::async_copy_host_to_device(partial_products_dev, partial_products, reduction_stream);
  combine<T>(products, reduction_stream, partial_products_dev);
  co_await xendv::await_stream(reduction_stream);
}

//--------------------------------------------------------------------------------------------------
// complete_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> complete_multiexponentiation(basct::span<T> res, unsigned element_num_bytes,
                                            basct::cspan<T> partial_products) noexcept {
  auto num_outputs = res.size();
  auto num_products = num_outputs * element_num_bytes * 8u;

  basdv::stream stream;
  memr::async_device_resource resource{stream};
#if 0
  basl::info("reducing {} products to {} outputs", num_products, num_products);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  reduce_products<T>(res_dev, stream, output_bit_table, products);
  products.reset();
#endif

  // combine the partial results
  memmg::managed_array<T> partial_products_dev{partial_products.size(), &resource};
  basdv::async_copy_host_to_device(partial_products_dev, partial_products, stream);
  memmg::managed_array<T> products{num_products, &resource};
  combine<T>(products, stream, partial_products_dev);
  partial_products_dev.reset();

  // reduce the products
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  reduce_products<T>(res_dev, stream, products);
  products.reset();

  // copy result
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> multiexponentiate_impl(basct::span<T> res,
                                      const partition_table_accessor<U>& accessor,
                                      unsigned element_num_bytes, basct::cspan<uint8_t> scalars,
                                      const multiexponentiate_options& options) noexcept {
  auto num_outputs = res.size();
  auto n = scalars.size() / (num_outputs * element_num_bytes);
  auto num_products = num_outputs * element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % (num_outputs * element_num_bytes) == 0
      // clang-format on
  );

  // compute bitwise products
  //
  // We split the work by groups of generators so that a single chunk will process
  // all the outputs for those generators. This minimizes the amount of host->device
  // copying we need to do for the table of precomputed sums.
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}
                                                    .chunk_multiple(16)
                                                    .min_chunk_size(options.min_chunk_size)
                                                    .max_chunk_size(options.max_chunk_size),
                                                options.split_factor);
  auto num_chunks = std::distance(chunk_first, chunk_last);
  if (num_chunks == 1) {
    multiexponentiate_no_chunks(res, accessor, element_num_bytes, scalars);
    co_return;
  }

  memmg::managed_array<T> products{num_products * num_chunks, memr::get_pinned_resource()};
  size_t chunk_index = 0;
  basl::info("computing {} bitwise multiexponentiation products of length {} using {} chunks",
             num_products, n, num_chunks);
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        basl::info("computing {} multiproducts for generators [{}, {}] on device {}", num_products,
                   rng.a(), rng.b(), basdv::get_device());
        memmg::managed_array<T> products_dev{num_products, memr::get_device_resource()};
        auto scalars_slice = scalars.subspan(num_outputs * element_num_bytes * rng.a(),
                                             rng.size() * num_outputs * element_num_bytes);
        co_await async_partition_product<T>(products_dev, accessor, scalars_slice, rng.a());
        basdv::stream stream;
        basdv::async_copy_device_to_host(
            basct::subspan(products, num_products * chunk_index, num_products), products_dev,
            stream);
        ++chunk_index;
        co_await xendv::await_stream(stream);
      });

  // complete the multi-exponentiation
  basl::info("reducing products for {} outputs", num_outputs);
  co_await complete_multiexponentiation<T>(res, element_num_bytes, products);
  basl::info("completed {} reductions", num_outputs);
}

template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
multiexponentiate_impl(basct::span<T> res, const partition_table_accessor<U>& accessor,
                       basct::cspan<unsigned> output_bit_table, basct::cspan<uint8_t> scalars,
                       const multiexponentiate_options& options) noexcept {
  auto num_outputs = res.size();
  auto num_products = std::accumulate(output_bit_table.begin(), output_bit_table.end(), 0u);
  auto num_output_bytes = basn::divide_up<size_t>(num_products, 8);
  auto n = scalars.size() / num_output_bytes;
#if 0
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % num_products == 0
      // clang-format on
  );
#endif

  // compute bitwise products
  //
  // We split the work by groups of generators so that a single chunk will process
  // all the outputs for those generators. This minimizes the amount of host->device
  // copying we need to do for the table of precomputed sums.
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}
                                                    .chunk_multiple(16)
                                                    .min_chunk_size(options.min_chunk_size)
                                                    .max_chunk_size(options.max_chunk_size),
                                                options.split_factor);
  (void)chunk_first;
  (void)chunk_last;
  auto num_chunks = std::distance(chunk_first, chunk_last);
  if (num_chunks == 1) {
    multiexponentiate_no_chunks(res, accessor, output_bit_table, num_products, scalars);
    co_return;
  }

  baser::panic("not implemented yet");
#if 0
  memmg::managed_array<T> products{num_products * num_chunks, memr::get_pinned_resource()};
  size_t chunk_index = 0;
  basl::info("computing {} bitwise multiexponentiation products of length {} using {} chunks",
             num_products, n, num_chunks);
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        basl::info("computing {} multiproducts for generators [{}, {}] on device {}", num_products,
                   rng.a(), rng.b(), basdv::get_device());
        memmg::managed_array<T> products_dev{num_products, memr::get_device_resource()};
        auto scalars_slice = scalars.subspan(num_outputs * element_num_bytes * rng.a(),
                                             rng.size() * num_outputs * element_num_bytes);
        co_await async_partition_product<T>(products_dev, accessor, scalars_slice, rng.a());
        basdv::stream stream;
        basdv::async_copy_device_to_host(
            basct::subspan(products, num_products * chunk_index, num_products), products_dev,
            stream);
        ++chunk_index;
        co_await xendv::await_stream(stream);
      });

  // complete the multi-exponentiation by splitting the remaining work by output
  auto [output_first, output_last] =
      basit::split(basit::index_range{0, num_outputs}, options.split_factor);
  basl::info("reducing products for {} outputs using {} chunks", num_outputs,
             std::distance(output_first, output_last));
  co_await xendv::concurrent_for_each(
      output_first, output_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        basl::info("reducing products for outputs [{}, {}] on device {}", rng.a(), rng.b(),
                   basdv::get_device());
        co_await complete_multiexponentiation<T>(res.subspan(rng.a(), rng.size()),
                                                 element_num_bytes, products, num_products,
                                                 rng.a() * element_num_bytes * 8u);
      });
#endif
}

//--------------------------------------------------------------------------------------------------
// async_multiexponentiate
//--------------------------------------------------------------------------------------------------
/**
 * Compute a multi-exponentiation using an accessor to precompute sums of partition groups.
 *
 * This implements the partition part of Pipenger's algorithm. See Algorithm 7 of
 * https://cacr.uwaterloo.ca/techreports/2010/cacr2010-26.pdf
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
async_multiexponentiate(basct::span<T> res, const partition_table_accessor<U>& accessor,
                        unsigned element_num_bytes, basct::cspan<uint8_t> scalars) noexcept {
  multiexponentiate_options options;
  options.split_factor = static_cast<unsigned>(basdv::get_num_devices());
  return multiexponentiate_impl(res, accessor, element_num_bytes, scalars, options);
}

template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_multiexponentiate(basct::span<T> res,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<unsigned> output_bit_table,
                                       basct::cspan<uint8_t> scalars) noexcept {
  multiexponentiate_options options;
  options.split_factor = static_cast<unsigned>(basdv::get_num_devices());
  return multiexponentiate_impl(res, accessor, output_bit_table, scalars, options);
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
/**
 * Host version of async_multiexponentiate.
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
void multiexponentiate(basct::span<T> res, const partition_table_accessor<U>& accessor,
                       unsigned element_num_bytes, basct::cspan<uint8_t> scalars) noexcept {
  auto num_outputs = res.size();
  auto n = scalars.size() / (num_outputs * element_num_bytes);
  auto num_products = num_outputs * element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % (num_outputs * element_num_bytes) == 0
      // clang-format on
  );

  // compute bitwise products
  basl::info("computing {} bitwise multiexponentiation products of length {}", num_products, n);
  memmg::managed_array<T> products(num_products);
  partition_product<T>(products, accessor, scalars, 0);

  // reduce products
  basl::info("reducing {} products to {} outputs", num_products, num_outputs);
  reduce_products<T>(res, products);
  basl::info("completed {} reductions", num_outputs);
}
} // namespace sxt::mtxpp2
