#include "sxt/base/device/pinned_buffer2.h"

#include "sxt/base/device/pinned_buffer_pool.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
pinned_buffer2::pinned_buffer2(pinned_buffer2&& other) noexcept
    : handle_{std::exchange(other.handle_, nullptr)}, size_{std::exchange(other.size_, 0)} {}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_buffer2::~pinned_buffer2() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  get_pinned_buffer_pool()->release_handle(handle_);
}

//--------------------------------------------------------------------------------------------------
// capacity
//--------------------------------------------------------------------------------------------------
size_t pinned_buffer2::capacity() const noexcept { return pinned_buffer_size; }

//--------------------------------------------------------------------------------------------------
// fill
//--------------------------------------------------------------------------------------------------
basct::cspan<std::byte> pinned_buffer2::fill_from_host(basct::cspan<std::byte> src) noexcept {
  if (src.empty()) {
    return src;
  }
  if (handle_ == nullptr) {
    handle_ = get_pinned_buffer_pool()->acquire_handle();
  }
  auto n = std::min(src.size(), this->capacity() - size_);
  std::copy_n(src.data(), n, static_cast<std::byte*>(handle_->ptr) + size_);
  size_ += n;
  return {src.data() + n, src.size() - n};
}
} // namespace sxt::basdv
