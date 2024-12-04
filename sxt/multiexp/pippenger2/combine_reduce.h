#pragma once

#include <numeric>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/strided_copy.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// combine_reduce_chunk
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
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
  memmg::managed_array<T> partials_dev{slice_num_partials * reduction_size, &resource};
  co_await xendv::strided_copy_host_to_device(partials_dev, stream, partial_products, num_partials,
                                              slice_num_partials, partials_offset);
  /* template <class T> */
  /* xena::future<> strided_copy_host_to_device(basct::span<T> dst, const basdv::stream& stream, */
  /*                                            basct::cspan<T> src, size_t stride, size_t
   * slice_size, */
  /*                                            size_t offset) noexcept { */
}
#pragma clang diagnostic pop

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

  // split
  auto [chunk_first, chunk_last] =
      basit::split(basit::index_range{0, num_outputs}, basdv::get_num_devices());

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
