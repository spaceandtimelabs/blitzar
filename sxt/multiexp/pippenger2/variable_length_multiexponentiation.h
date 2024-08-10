#pragma once

#include "sxt/multiexp/pippenger2/multiexponentiation.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// async_partition_product_chunk
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
async_partition_product_chunk(basct::span<T> products, const partition_table_accessor<U>& accessor,
                              basct::cspan<unsigned> output_bit_table,
                              basct::cspan<unsigned> output_lengths, basct::cspan<uint8_t> scalars,
                              unsigned first, unsigned length) noexcept {
  auto num_products = products.size();

  // product lengths
  memmg::managed_array<unsigned> product_lengths_data{num_products, memr::get_pinned_resource()};
  basct::span<unsigned> product_lengths{product_lengths_data};
  compute_product_length_table(product_lengths, output_bit_table, output_lengths, first, length);

  // launch kernel
  auto num_products_p = product_lengths.size();
  SXT_DEBUG_ASSERT(num_products_p <= num_products);
  auto products_fut = [&]() noexcept -> xena::future<> {
    if (num_products_p > 0) {
      return async_partition_product(products.subspan(num_products - num_products_p), num_products,
                                     accessor, scalars, product_lengths, first);
    } else {
      return xena::make_ready_future();
    }
  }();

  // fill in zero section
  memmg::managed_array<T> identities_host{num_products - num_products_p,
                                          memr::get_pinned_resource()};
  std::fill(identities_host.begin(), identities_host.end(), T::identity());
  basdv::stream stream;
  basdv::async_copy_host_to_device(products.subspan(0, num_products - num_products_p),
                                   identities_host, stream);

  // await futures
  co_await xendv::await_stream(stream);
  co_await std::move(products_fut);
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate_product_step
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<>
multiexponentiate_product_step(basct::span<T> products, basdv::stream& reduction_stream,
                               const partition_table_accessor<U>& accessor,
                               unsigned num_output_bytes, basct::cspan<unsigned> output_bit_table,
                               basct::cspan<unsigned> output_lengths, basct::cspan<uint8_t> scalars,
                               const multiexponentiate_options& options) noexcept {
  auto num_products = products.size();
  auto n = scalars.size() / num_output_bytes;
  auto window_width = accessor.window_width();

  // compute bitwise products
  //
  // We split the work by groups of generators so that a single chunk will process
  // all the outputs for those generators. This minimizes the amount of host->device
  // copying we need to do for the table of precomputed sums.
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}
                                                    .chunk_multiple(window_width)
                                                    .min_chunk_size(options.min_chunk_size)
                                                    .max_chunk_size(options.max_chunk_size),
                                                options.split_factor);
  auto num_chunks = static_cast<size_t>(std::distance(chunk_first, chunk_last));
  basl::info("computing {} bitwise multiexponentiation products of length {} using {} chunks",
             num_products, n, num_chunks);

  // handle no chunk case
  if (num_chunks == 1) {
    co_await async_partition_product_chunk(products, accessor, output_bit_table, output_lengths,
                                           scalars, 0, n);
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
        co_await async_partition_product_chunk<T>(partial_products_dev, accessor, output_bit_table,
                                                  output_lengths, scalars_slice, rng.a(),
                                                  rng.size());
        basdv::stream stream;
        basdv::async_copy_device_to_host(
            basct::subspan(partial_products, num_products * chunk_index, num_products),
            partial_products_dev, stream);
        ++chunk_index;
        co_await xendv::await_stream(stream);
      });

  // combine the partial products
  basl::info("combining {} partial product chunks", num_chunks);
  memr::async_device_resource resource{reduction_stream};
  memmg::managed_array<T> partial_products_dev{partial_products.size(), &resource};
  basdv::async_copy_host_to_device(partial_products_dev, partial_products, reduction_stream);
  combine<T>(products, reduction_stream, partial_products_dev);
  co_await xendv::await_stream(reduction_stream);
}
//--------------------------------------------------------------------------------------------------
// async_multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_multiexponentiate(basct::span<T> res,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<unsigned> output_bit_table,
                                       basct::cspan<unsigned> output_lengths,
                                       basct::cspan<uint8_t> scalars) noexcept {
  multiexponentiate_options options;
  options.split_factor = static_cast<unsigned>(basdv::get_num_devices());
  return multiexponentiate_impl(res, accessor, output_bit_table, output_lengths, scalars, options);
}
} // namespace sxt::mtxpp2
