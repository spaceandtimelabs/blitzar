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
#include "sxt/memory/resource/chained_resource.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
chained_resource::chained_resource() noexcept
    : chained_resource{std::pmr::get_default_resource()} {}

chained_resource::chained_resource(std::pmr::memory_resource* upstream) noexcept
    : upstream_{upstream} {}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
chained_resource::~chained_resource() noexcept {
  for (auto [ptr, bytes, alignment] : allocations_) {
    upstream_->deallocate(ptr, bytes, alignment);
  }
}

//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* chained_resource::do_allocate(size_t bytes, size_t alignment) noexcept {
  auto ptr = upstream_->allocate(bytes, alignment);
  allocations_.emplace_back(ptr, bytes, alignment);
  return ptr;
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool chained_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}
} // namespace sxt::memr
