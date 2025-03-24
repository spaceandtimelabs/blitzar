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

#include "sxt/base/device/property.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// plan_split
//--------------------------------------------------------------------------------------------------
basit::split_options plan_split(size_t bytes) noexcept {
  auto device_memory = get_total_device_memory();

  auto high_memory_target = device_memory / 16u;
  auto low_memory_target = device_memory / 64u;
  /* auto high_memory_target = device_memory / (100u * 16u); */
  /* auto low_memory_target = device_memory / (100u * 64u); */

  auto high_target = high_memory_target / bytes;
  auto low_target = low_memory_target / bytes;

  high_target = std::max<size_t>(1u, high_target);
  low_target = std::max<size_t>(1u, low_target);

  return basit::split_options{
      .min_chunk_size = low_target,
      .max_chunk_size = high_target,
      .split_factor = basdv::get_num_devices() * 2u,
  };
}
} // namespace sxt::basdv
