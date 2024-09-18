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

#include <memory_resource>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// pinned_resource
//--------------------------------------------------------------------------------------------------
class pinned_resource final : public std::pmr::memory_resource {
public:
private:
  void* do_allocate(size_t bytes, size_t alignment) noexcept override;

  void do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_pinned_resource
//--------------------------------------------------------------------------------------------------
std::pmr::memory_resource* get_pinned_resource() noexcept;
} // namespace sxt::memr
