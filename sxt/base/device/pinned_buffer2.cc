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
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
pinned_buffer2& pinned_buffer2::operator=(pinned_buffer2&& other) noexcept {
  this->reset();
  handle_ = std::exchange(other.handle_, nullptr);
  size_ = std::exchange(other.size_, 0);
  return *this;
}

//--------------------------------------------------------------------------------------------------
// capacity
//--------------------------------------------------------------------------------------------------
size_t pinned_buffer2::capacity() noexcept { return pinned_buffer_size; }

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
  return src.subspan(n);
}

//--------------------------------------------------------------------------------------------------
// reset
//--------------------------------------------------------------------------------------------------
void pinned_buffer2::reset() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  get_pinned_buffer_pool()->release_handle(handle_);
  handle_ = nullptr;
  size_ = 0;
}
} // namespace sxt::basdv
