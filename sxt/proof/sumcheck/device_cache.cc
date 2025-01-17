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
#include "sxt/proof/sumcheck/device_cache.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/memory/resource/device_resource.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// make_device_copy
//--------------------------------------------------------------------------------------------------
static std::unique_ptr<device_cache_data>
make_device_copy(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, basdv::stream& stream) noexcept {
  device_cache_data res{
      .product_table{product_table.size(), memr::get_device_resource()},
      .product_terms{product_terms.size(), memr::get_device_resource()},
  };
  basdv::async_copy_host_to_device(res.product_table, product_table, stream);
  basdv::async_copy_host_to_device(res.product_terms, product_terms, stream);
  return std::make_unique<device_cache_data>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
device_cache::device_cache(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms) noexcept
    : product_table_{product_table}, product_terms_{product_terms} {}

//--------------------------------------------------------------------------------------------------
// lookup
//--------------------------------------------------------------------------------------------------
void device_cache::lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table,
                          basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept {
  auto& ptr = data_[basdv::get_device()];
  if (ptr == nullptr) {
    ptr = make_device_copy(product_table_, product_terms_, stream);
  }
  product_table = ptr->product_table;
  product_terms = ptr->product_terms;
}

//--------------------------------------------------------------------------------------------------
// clear
//--------------------------------------------------------------------------------------------------
std::unique_ptr<device_cache_data> device_cache::clear() noexcept {
  auto res{std::move(data_[basdv::get_device()])};
  for (auto& ptr : data_) {
    ptr.reset();
  }
  return res;
}
} // namespace sxt::prfsk
