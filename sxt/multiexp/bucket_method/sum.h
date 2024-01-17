#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/multiexp/bucket_method/bucket_descriptor.h"
#include "sxt/multiexp/bucket_method/multiproduct_table.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// compute_bucket_sums 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators,
                                   basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                                   unsigned n, unsigned bit_width) noexcept {
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<bucket_descriptor> table{&resource};
  memmg::managed_array<unsigned> indexes{&resource};
  auto fut = compute_multiproduct_table(table, indexes, scalars, element_num_bytes, n, bit_width);

  co_await std::move(fut);
  (void)table;
  (void)indexes;
/* xena::future<> compute_multiproduct_table(memmg::managed_array<bucket_descriptor>& table, */
/*                                           memmg::managed_array<unsigned>& indexes, */
/*                                           basct::cspan<const uint8_t*> scalars, */
/*                                           unsigned element_num_bytes, unsigned n, */
/*                                           unsigned bit_width) noexcept; */
  (void)resource;
  (void)sums;
  (void)generators;
  (void)scalars;
  (void)element_num_bytes;
  (void)n;
  (void)bit_width;
}
} // namespace sxt::mtxbk
