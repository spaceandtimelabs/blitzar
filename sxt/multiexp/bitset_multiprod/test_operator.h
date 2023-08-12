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
#include <cstdint>

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// test_operator
//--------------------------------------------------------------------------------------------------
class test_operator {
public:
  test_operator(size_t* counter) noexcept : counter_{counter} {}

  void mark_unset(uint64_t& value) const noexcept { value = 0; }

  bool is_set(uint64_t value) const noexcept { return value != 0; }

  void add_bitwise_entries(uint64_t& res, uint64_t lhs, uint64_t rhs) const noexcept {
    ++*counter_;
    res = lhs + rhs;
  }

private:
  size_t* counter_;
};
} // namespace sxt::mtxbmp
