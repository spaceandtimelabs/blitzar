#pragma once

#include "sxt/base/device/pinned_buffer_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer_ptr
//--------------------------------------------------------------------------------------------------
class pinned_buffer_ptr {
 public:
   pinned_buffer_ptr() noexcept;
   pinned_buffer_ptr(pinned_buffer_ptr&& ptr) noexcept;
   pinned_buffer_ptr(const pinned_buffer_ptr&) noexcept = delete;

   ~pinned_buffer_ptr() noexcept;

   pinned_buffer_ptr& operator=(pinned_buffer_ptr&& ptr) noexcept;
   pinned_buffer_ptr& operator=(const pinned_buffer_ptr& ptr) noexcept = delete;

   operator void*() noexcept {
     return handle_->ptr;
   }

   operator const void*() const noexcept {
     return handle_->ptr;
   }
 private:
   pinned_buffer_handle* handle_;
};
} // namespace sxt::basdv
