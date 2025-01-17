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
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::basdv {
class stream;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// device_cache_data
//--------------------------------------------------------------------------------------------------
struct device_cache_data {
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
};

//--------------------------------------------------------------------------------------------------
// device_cache
//--------------------------------------------------------------------------------------------------
class device_cache {
public:
  device_cache(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
               basct::cspan<unsigned> product_terms) noexcept;

  void lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table,
              basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept;

  std::unique_ptr<device_cache_data> clear() noexcept;

private:
  basct::cspan<std::pair<s25t::element, unsigned>> product_table_;
  basct::cspan<unsigned> product_terms_;
  basdv::device_map<std::unique_ptr<device_cache_data>> data_;
};
} // namespace sxt::prfsk
