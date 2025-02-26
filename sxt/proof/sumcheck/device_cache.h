/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

#include <utility>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/device/device_map.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/field/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// device_cache_data
//--------------------------------------------------------------------------------------------------
template <basfld::element T> struct device_cache_data {
  memmg::managed_array<std::pair<T, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
};

//--------------------------------------------------------------------------------------------------
// make_device_copy
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
std::unique_ptr<device_cache_data<T>>
make_device_copy(basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, basdv::stream& stream) noexcept {
  device_cache_data<T> res{
      .product_table{product_table.size(), memr::get_device_resource()},
      .product_terms{product_terms.size(), memr::get_device_resource()},
  };
  basdv::async_copy_host_to_device(res.product_table, product_table, stream);
  basdv::async_copy_host_to_device(res.product_terms, product_terms, stream);
  return std::make_unique<device_cache_data<T>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// device_cache
//--------------------------------------------------------------------------------------------------
template <basfld::element T> class device_cache {
public:
  device_cache(basct::cspan<std::pair<T, unsigned>> product_table,
               basct::cspan<unsigned> product_terms) noexcept
      : product_table_{product_table}, product_terms_{product_terms} {}

  void lookup(basct::cspan<std::pair<T, unsigned>>& product_table,
              basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept {
    auto& ptr = data_[basdv::get_device()];
    if (ptr == nullptr) {
      ptr = make_device_copy(product_table_, product_terms_, stream);
    }
    product_table = ptr->product_table;
    product_terms = ptr->product_terms;
  }

  std::unique_ptr<device_cache_data<T>> clear() noexcept {
    auto res{std::move(data_[basdv::get_device()])};
    for (auto& ptr : data_) {
      ptr.reset();
    }
    return res;
  }

private:
  basct::cspan<std::pair<T, unsigned>> product_table_;
  basct::cspan<unsigned> product_terms_;
  basdv::device_map<std::unique_ptr<device_cache_data<T>>> data_;
};
} // namespace sxt::prfsk2
