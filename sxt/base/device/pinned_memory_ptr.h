#pragma once

#include "sxt/base/device/pinned_memory_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_memory_ptr
//--------------------------------------------------------------------------------------------------
class pinned_memory_ptr {
 public:
   pinned_memory_ptr() noexcept;
   pinned_memory_ptr(pinned_memory_ptr&& ptr) noexcept;
   pinned_memory_ptr(const pinned_memory_ptr&) noexcept = delete;

   ~pinned_memory_ptr() noexcept;

   pinned_memory_ptr& operator=(pinned_memory_ptr&& ptr) noexcept;
   pinned_memory_ptr& operator=(const pinned_memory_ptr& ptr) noexcept = delete;

   operator void*() noexcept {
     return handle_->ptr;
   }

   operator const void*() const noexcept {
     return handle_->ptr;
   }
 private:
   pinned_memory_handle* handle_;
};
} // namespace sxt::basdv
