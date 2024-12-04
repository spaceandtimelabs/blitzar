#pragma once

#include "sxt/base/device/pinned_buffer_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer
//--------------------------------------------------------------------------------------------------
class pinned_buffer {
 public:
   pinned_buffer() noexcept;
   pinned_buffer(pinned_buffer&& ptr) noexcept;
   pinned_buffer(const pinned_buffer&) noexcept = delete;

   ~pinned_buffer() noexcept;

   pinned_buffer& operator=(pinned_buffer&& ptr) noexcept;
   pinned_buffer& operator=(const pinned_buffer& ptr) noexcept = delete;

   static size_t size() noexcept;

   void* data() noexcept {
     return handle_->ptr;
   }

   const void* data() const noexcept {
     return handle_->ptr;
   }

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
