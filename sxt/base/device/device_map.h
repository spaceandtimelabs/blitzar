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

#include "sxt/base/device/property.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/max_devices.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// device_map
//--------------------------------------------------------------------------------------------------
template <class T> class device_map {
public:
  int size() const noexcept { return get_num_devices(); }

  T& operator[](int device) noexcept {
    SXT_DEBUG_ASSERT(0 <= device && device < this->size());
    return data_[device];
  }

  const T& operator[](int device) const noexcept {
    SXT_DEBUG_ASSERT(0 <= device && device < this->size());
    return data_[device];
  }

  T* begin() noexcept { return data_; }

  T* end() noexcept { return data_ + this->size(); }

  const T* begin() const noexcept { return data_; }

  const T* end() const noexcept { return data_ + this->size(); }

private:
  T data_[SXT_MAX_DEVICES] = {};
};
} // namespace sxt::basdv
