/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// active_device_guard
//--------------------------------------------------------------------------------------------------
class active_device_guard {
public:
  active_device_guard() noexcept;

  explicit active_device_guard(unsigned device) noexcept
      : active_device_guard{static_cast<int>(device)} {}

  explicit active_device_guard(int device) noexcept;

  ~active_device_guard() noexcept;

private:
  int prev_device_;
};
} // namespace sxt::basdv
