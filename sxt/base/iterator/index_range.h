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

#include <cstddef>
#include <limits>

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// index_range
//--------------------------------------------------------------------------------------------------
class index_range {
public:
  index_range() noexcept = default;

  index_range(size_t a, size_t b) noexcept;

  index_range(size_t a, size_t b, size_t chunk_multiple) noexcept;

  size_t a() const noexcept { return a_; }
  size_t b() const noexcept { return b_; }

  size_t size() const noexcept { return b_ - a_; }

  bool operator==(const index_range&) const noexcept = default;

  size_t chunk_multiple() const noexcept { return chunk_multiple_; }

  [[nodiscard]] index_range chunk_multiple(size_t val) const noexcept;

private:
  size_t a_{0};
  size_t b_{0};
  size_t chunk_multiple_{1};
};
} // namespace sxt::basit
