#pragma once

#include "sxt/base/device/pinned_buffer_handle.h"

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer2
//--------------------------------------------------------------------------------------------------
class pinned_buffer2 {
public:
  pinned_buffer2() noexcept = default;
  pinned_buffer2(const pinned_buffer2&) noexcept = delete;
  pinned_buffer2(pinned_buffer2&& other) noexcept;

  ~pinned_buffer2() noexcept;

  pinned_buffer2& operator=(const pinned_buffer2&) noexcept = delete;
  pinned_buffer2& operator=(pinned_buffer2&& other) noexcept;

  bool empty() const noexcept {
    return size_ == 0;
  }

  bool full() const noexcept {
    return size_ == this->capacity();
  }

  size_t size() const noexcept {
    return size_;
  }

  static size_t capacity() noexcept;

  void* data() noexcept {
    if (handle_ == nullptr) {
      return nullptr;
    }
    return handle_->ptr;
  }

  const void* data() const noexcept {
    if (handle_ == nullptr) {
      return nullptr;
    }
    return handle_->ptr;
  }

  basct::cspan<std::byte> fill_from_host(basct::cspan<std::byte> src) noexcept;

  void reset() noexcept;

private:
  pinned_buffer_handle* handle_ = nullptr;
  size_t size_ = 0;
};
} // namespace sxt::basdv
