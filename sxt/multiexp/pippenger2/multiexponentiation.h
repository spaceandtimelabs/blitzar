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

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_product.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, const partition_table_accessor<T>& accessor,
                                 unsigned element_num_bytes,
                                 basct::cspan<uint8_t> scalars) noexcept {
  auto num_outputs = res.size();
  auto num_products = num_outputs * element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() % (num_outputs * element_num_bytes) == 0
      // clang-format on
  );
  
  // compute bitwise products
  memmg::managed_array<T> products(num_products, memr::get_device_resource());
  co_await partition_product<T>(products, accessor, scalars, 0);

  // reduce products
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> res_dev{num_outputs, stream};
  reduce_products<T>(res_dev, stream, products);

  // copy result
  products.reset();
  basdv::async_copy_device_to_host(res, stream, res_dev);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxpp2
