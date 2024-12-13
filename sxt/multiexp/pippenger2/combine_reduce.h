#pragma once

#include <cassert>
#include <numeric>

#include <print>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/split.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/copy.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// combine_reduce_chunk_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__device__ void combine_reduce_chunk_kernel(T* __restrict__ res, const T* __restrict__ partials,
                                            const unsigned* __restrict__ bit_table_partial_sums,
                                            unsigned num_partials, unsigned reduction_size,
                                            unsigned partials_offset,
                                            unsigned output_index) noexcept {
  auto output_index_p = umax(output_index, 1u) - 1u;
  auto output_correction = bit_table_partial_sums[output_index_p] * (output_index != 0) +
                               partials_offset * (output_index == 0);
  auto bit_width = bit_table_partial_sums[output_index] - output_correction;
  assert(bit_width > 0);

  // adjust points
  res += output_index;
  partials += bit_table_partial_sums[output_index] - partials_offset;

  // combine reduce
  unsigned bit_index = bit_width - 1u;
  --partials;
  T e = *partials;
  for (unsigned reduction_index=1; reduction_index<reduction_size; ++reduction_index) {
    auto ep = partials[reduction_index*num_partials];
    add_inplace(e, ep);
  }
  for (; bit_index-- > 0u;) {
    --partials;
    double_element(e, e);
    for (unsigned reduction_index = 0; reduction_index < reduction_size; ++reduction_index) {
      auto ep = partials[reduction_index * num_partials];
      add_inplace(e, ep);
    }
  }
  *res = e;
}

//--------------------------------------------------------------------------------------------------
// combine_reduce_chunk
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> combine_reduce_chunk(basct::span<T> res,
                                    basct::cspan<unsigned> output_bit_table_partial_sums,
                                    basct::cspan<T> partial_products, unsigned reduction_size,
                                    unsigned partials_offset) noexcept {
  auto num_partials = partial_products.size() / reduction_size;
  auto num_outputs = output_bit_table_partial_sums.size();
  unsigned slice_num_partials = output_bit_table_partial_sums[num_outputs - 1] - partials_offset;
  basdv::stream stream;

  // copy data
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partials_dev_data{&resource};
  basct::cspan<T> partials_dev = partial_products;
  if (!basdv::is_active_device_pointer(partials_dev.data())) {
    partials_dev_data.resize(slice_num_partials * reduction_size);
    co_await xendv::strided_copy_host_to_device<T>(partials_dev_data, stream, partial_products,
                                                   num_partials, slice_num_partials,
                                                   partials_offset);
    partials_dev = partials_dev_data;
  } else {
    SXT_RELEASE_ASSERT(partial_products.size() == slice_num_partials * reduction_size);
  }
  memmg::managed_array<unsigned> bit_table_partial_sums_dev{num_outputs, &resource};
  basdv::async_copy_host_to_device(bit_table_partial_sums_dev, output_bit_table_partial_sums, stream);

  // combine reduce chunk
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  auto f = [
               // clang-format off
    num_partials = slice_num_partials,
    reduction_size = reduction_size,
    partials_offset = partials_offset,
    res = res_dev.data(),
    partials = partials_dev.data(),
    bit_table_partial_sums = bit_table_partial_sums_dev.data()
               // clang-format on
  ] __device__
           __host__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             combine_reduce_chunk_kernel(res, partials, bit_table_partial_sums, num_partials,
                                         reduction_size, partials_offset, output_index);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// combine_reduce 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> combine_reduce(basct::span<T> res, basct::cspan<unsigned> output_bit_table,
                              basct::cspan<T> partial_products) noexcept {
  auto num_outputs = output_bit_table.size();

  // partials
  memmg::managed_array<unsigned> bit_table_partial_sums(num_outputs);
  std::partial_sum(output_bit_table.begin(), output_bit_table.end(),
                   bit_table_partial_sums.begin());
  auto reduction_size = partial_products.size() / bit_table_partial_sums[num_outputs - 1];

  // don't split if partials are already in device memory
  if (basdv::is_active_device_pointer(partial_products.data())) {
    co_return co_await combine_reduce_chunk(res, bit_table_partial_sums, partial_products,
                                            reduction_size, 0);
  }

  // split
  basit::split_options split_options{
    .max_chunk_size = 1024,
    .split_factor = basdv::get_num_devices(),
  };
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, num_outputs}, split_options);

  // combine reduce
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](basit::index_range rng) noexcept -> xena::future<> {
        auto output_first = rng.a();

        auto res_chunk = res.subspan(output_first, rng.size());
        auto bit_table_partial_sums_chunk =
            basct::subspan(bit_table_partial_sums, output_first, rng.size());
        auto partials_offset = output_first > 0 ? bit_table_partial_sums[output_first - 1] : 0u;

        co_await combine_reduce_chunk(res_chunk, bit_table_partial_sums_chunk, partial_products,
                                      reduction_size, partials_offset);
      });
}
} // namespace sxt::mtxpp2
