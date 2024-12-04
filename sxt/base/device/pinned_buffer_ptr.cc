#include "sxt/base/device/pinned_buffer_ptr.h"

#include "sxt/base/device/pinned_buffer_pool.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// consructor
//--------------------------------------------------------------------------------------------------
pinned_buffer_ptr::pinned_buffer_ptr() noexcept
    : handle_{get_pinned_buffer_pool()->aquire_handle()} {}

pinned_buffer_ptr::pinned_buffer_ptr(pinned_buffer_ptr&& ptr) noexcept : handle_{ptr.handle_} {
  ptr.handle_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_buffer_ptr::~pinned_buffer_ptr() noexcept {
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
pinned_buffer_ptr& pinned_buffer_ptr::operator=(pinned_buffer_ptr&& ptr) noexcept {
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
  handle_ = ptr.handle_;
  ptr.handle_ = nullptr;
  return *this;
}
} // namespace sxt::basdv
