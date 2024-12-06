#pragma once

#include <cstddef>

namespace sxt::basdv {
struct pinned_buffer_handle;

constexpr unsigned pinned_buffer_size = 1024u * 1024u * 2u; // 2 megabytes

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

  pinned_buffer_handle* aquire_handle() noexcept;

  void release_handle(pinned_buffer_handle* handle) noexcept;

  size_t num_buffers() const noexcept;

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
