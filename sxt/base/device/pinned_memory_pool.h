#pragma once

#include <cstddef>

namespace sxt::basdv {
struct pinned_memory_handle;

constexpr unsigned pinned_memory_size = 1024u * 1024u * 2; // 2 megabytes

//--------------------------------------------------------------------------------------------------
// pinned_memory_pool
//--------------------------------------------------------------------------------------------------
class pinned_memory_pool {
 public:
  explicit pinned_memory_pool(size_t initial_size) noexcept;

  ~pinned_memory_pool() noexcept;

  pinned_memory_pool(const pinned_memory_pool&) = delete;
  pinned_memory_pool(pinned_memory_pool&&) = delete;
  pinned_memory_pool& operator=(const pinned_memory_pool&) = delete;

  pinned_memory_handle* aquire_handle() noexcept;

  void release_handle(pinned_memory_handle* handle) noexcept;

private:
  pinned_memory_handle* head_ = nullptr;
};
} // namespace sxt::basdv
