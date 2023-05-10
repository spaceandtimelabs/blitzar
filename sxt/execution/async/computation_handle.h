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
#pragma once

namespace sxt::basdv {
class stream;
}
namespace sxt::basdv {
struct stream_handle;
}

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// computation_handle
//--------------------------------------------------------------------------------------------------
class computation_handle {
public:
  computation_handle() noexcept = default;
  computation_handle(const computation_handle&) = delete;
  computation_handle(computation_handle&& other) noexcept;

  ~computation_handle() noexcept;

  computation_handle& operator=(const computation_handle&) = delete;
  computation_handle& operator=(computation_handle&& other) noexcept;

  void wait() noexcept;

  bool empty() const noexcept { return head_ == nullptr; }

  void add_stream(basdv::stream&& stream) noexcept;

private:
  basdv::stream_handle* head_{nullptr};
};
} // namespace sxt::xena
