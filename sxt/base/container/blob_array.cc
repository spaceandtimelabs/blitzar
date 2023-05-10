/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/base/container/blob_array.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
blob_array::blob_array(size_t size, size_t blob_size) noexcept
    : blob_size_{blob_size == 0 ? 1 : blob_size}, data_(size * blob_size_) {}

//--------------------------------------------------------------------------------------------------
// resize
//--------------------------------------------------------------------------------------------------
void blob_array::resize(size_t size, size_t blob_size) noexcept {
  blob_size_ = blob_size == 0 ? 1 : blob_size;
  data_.resize(size * blob_size_);
}
} // namespace sxt::basct
