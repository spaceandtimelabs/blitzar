#pragma once

#include <cstddef>
#include <type_traits>

#include "sxt/base/type/polymorphic_allocator.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::memmg {
//--------------------------------------------------------------------------------------------------
// managed_array
//--------------------------------------------------------------------------------------------------
// void
template <>
class managed_array<void> {
 public:
   using allocator_type = bast::polymorphic_allocator;

   // constructor
   managed_array() noexcept = default;

   managed_array(allocator_type alloc) noexcept : alloc_{alloc} {}

   managed_array(void* data, size_t size, size_t num_bytes,
                 allocator_type alloc = {}) noexcept;

   managed_array(const managed_array& other) noexcept;

   managed_array(const managed_array& other,
                 allocator_type alloc) noexcept;

   managed_array(managed_array&& other) noexcept;

   managed_array(managed_array&& other, allocator_type alloc) noexcept;

   // assignment
   managed_array& operator=(const managed_array& other) noexcept;
   managed_array& operator=(managed_array&& other) noexcept;

   // destructor
   ~managed_array() noexcept;

   // accessors
   void* data() const noexcept { return data_; }
   const void* data() noexcept { return data_; }

   size_t size() const noexcept { return size_; }

   allocator_type get_allocator() const noexcept { return alloc_; }

   // methods
   void reset() noexcept;
 private:
   allocator_type alloc_;
   void* data_{nullptr};
   size_t size_{0};
   size_t num_bytes_{0};
};

// general
template <class T>
class managed_array {
  static_assert(std::is_trivially_destructible_v<T>);
 public:
 private:
};
} // namespace sxt::memmg
