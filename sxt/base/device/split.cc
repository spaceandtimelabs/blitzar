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
#include "sxt/base/device/split.h"

#include <cassert>
#include <cmath>

#include "sxt/base/device/property.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// plan_split_impl
//--------------------------------------------------------------------------------------------------
basit::split_options plan_split_impl(size_t bytes, size_t total_device_memory,
                                     double memory_target_low, double memory_target_high,
                                     size_t split_factor) noexcept {
  assert(memory_target_low <= memory_target_high);
  auto target_low =
      static_cast<size_t>(std::floor(total_device_memory * memory_target_low / bytes));
  target_low = std::max(size_t{1}, target_low);
  auto target_high =
      static_cast<size_t>(std::floor(total_device_memory * memory_target_high / bytes));
  target_high = std::max(size_t{1}, target_high);
  return basit::split_options{
      .min_chunk_size = target_low,
      .max_chunk_size = target_high,
      .split_factor = split_factor,
  };
}

//--------------------------------------------------------------------------------------------------
// plan_split
//--------------------------------------------------------------------------------------------------
basit::split_options plan_split(size_t bytes) noexcept {
  auto memory_target_low = 1.0 / 64.0;
  auto memory_target_high = 1.0 / 16.0;
  auto total_device_memory = get_total_device_memory();
  auto split_factor = basdv::get_num_devices() * 2u;
  return plan_split_impl(bytes, total_device_memory, memory_target_low, memory_target_high,
                         split_factor);
}
} // namespace sxt::basdv
