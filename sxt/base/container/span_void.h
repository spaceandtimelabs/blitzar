#pragma once

#include <cassert>
#include <cstddef>

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// span_void
//--------------------------------------------------------------------------------------------------
class span_void {
 public:
   span_void() noexcept = default;

   span_void(void* data, size_t size, size_t element_size) noexcept
       : size_{size}, element_size_{element_size}, data_{data} {}

   void* data() const noexcept { return data_; }

   bool empty() const noexcept { return size_ == 0; }

   size_t size() const noexcept { return size_; }

   span_void subspan(size_t offset) const noexcept {
     assert(offset <= size_);
     return {
         static_cast<void*>(static_cast<char*>(data_) + element_size_ * offset),
         size_ - offset,
         element_size_,
     };
   }

   span_void subspan(size_t offset, size_t size_p) const noexcept {
     assert(offset <= size_);
     assert(offset + size_p <= size_);
     return {
         static_cast<void*>(static_cast<char*>(data_) + element_size_ * offset),
         size_p,
         element_size_,
     };
   }

 private:
   size_t size_ = 0;
   size_t element_size_ = 0;
   void* data_ = nullptr;
};
} // namespace sxt::basct
