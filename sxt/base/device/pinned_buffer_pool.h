/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

namespace sxt::basdv {
struct pinned_buffer_handle;

/* constexpr unsigned pinned_buffer_size = 1024u * 1024u * 2u; // 2 megabytes */
constexpr unsigned pinned_buffer_size = 1024u * 1024u * 4u; // 2 megabytes

//--------------------------------------------------------------------------------------------------
// pinned_buffer_pool
//--------------------------------------------------------------------------------------------------
class pinned_buffer_pool {
public:
  explicit pinned_buffer_pool(size_t initial_size) noexcept;

  ~pinned_buffer_pool() noexcept;

  pinned_buffer_pool(const pinned_buffer_pool&) = delete;
  pinned_buffer_pool(pinned_buffer_pool&&) = delete;
  pinned_buffer_pool& operator=(const pinned_buffer_pool&) = delete;

  pinned_buffer_handle* acquire_handle() noexcept;

  void release_handle(pinned_buffer_handle* handle) noexcept;

  size_t size() const noexcept;

private:
  pinned_buffer_handle* head_ = nullptr;
};

//--------------------------------------------------------------------------------------------------
// get_pinned_buffer_pool
//--------------------------------------------------------------------------------------------------
/**
 * Access the thread_local pinned pool.
 */
pinned_buffer_pool* get_pinned_buffer_pool(size_t initial_size = 16) noexcept;
} // namespace sxt::basdv
