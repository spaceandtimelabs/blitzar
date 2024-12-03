#include "sxt/base/device/pinned_memory_ptr.h"

#include "sxt/base/device/pinned_memory_pool.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// consructor
//--------------------------------------------------------------------------------------------------
pinned_memory_ptr::pinned_memory_ptr() noexcept
    : handle_{get_pinned_memory_pool()->aquire_handle()} {}

pinned_memory_ptr::pinned_memory_ptr(pinned_memory_ptr&& ptr) noexcept : handle_{ptr.handle_} {
  ptr.handle_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_memory_ptr::~pinned_memory_ptr() noexcept {
  if (handle_ != nullptr) {
    get_pinned_memory_pool()->release_handle(handle_);
  }
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
pinned_memory_ptr& pinned_memory_ptr::operator=(pinned_memory_ptr&& ptr) noexcept {
  if (handle_ != nullptr) {
    get_pinned_memory_pool()->release_handle(handle_);
  }
  handle_ = ptr.handle_;
  ptr.handle_ = nullptr;
  return *this;
}
} // namespace sxt::basdv
