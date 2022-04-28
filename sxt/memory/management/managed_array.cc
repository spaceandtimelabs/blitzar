#include "sxt/memory/management/managed_array.h"

#include <cstring>

namespace sxt::memmg {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
managed_array<void>::managed_array(void* data, size_t size, size_t num_bytes,
                                   allocator_type alloc) noexcept
    : alloc_{alloc}, data_{data}, size_{size}, num_bytes_{num_bytes} {}

managed_array<void>::managed_array(const managed_array& other) noexcept
    : managed_array{other, other.get_allocator()} {}

managed_array<void>::managed_array(const managed_array& other,
                                   allocator_type alloc) noexcept
    : alloc_{alloc} {
  this->operator=(other);
}

managed_array<void>::managed_array(managed_array&& other) noexcept
    : managed_array{std::move(other), other.get_allocator()} {}

managed_array<void>::managed_array(managed_array&& other,
                                   allocator_type alloc) noexcept
    : alloc_{alloc} {
  this->operator=(std::move(other));
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
managed_array<void>::~managed_array() noexcept {
  this->reset();
}

//--------------------------------------------------------------------------------------------------
// assignment
//--------------------------------------------------------------------------------------------------
managed_array<void>& managed_array<void>::operator=(const managed_array& other) noexcept {
  if (this == &other) {
    return *this;
  }
  this->reset();
  num_bytes_ = other.num_bytes_;
  size_ = other.size_;
  data_ = static_cast<void*>(alloc_.allocate(num_bytes_));
  std::memcpy(data_, other.data_, num_bytes_);
  return *this;
}

managed_array<void>& managed_array<void>::operator=(managed_array&& other) noexcept {
  if (this->get_allocator() != other.get_allocator()) {
    return this->operator=(other);
  }
  this->reset();

  num_bytes_ = other.num_bytes_;
  size_ = other.size_;
  data_ = other.data_;

  other.num_bytes_ = 0;
  other.size_ = 0;
  other.data_ = nullptr;

  return *this;
}

//--------------------------------------------------------------------------------------------------
// reset
//--------------------------------------------------------------------------------------------------
void managed_array<void>::reset() noexcept {
  if (data_ == nullptr) {
    return;
  }
  alloc_.deallocate(reinterpret_cast<std::byte*>(data_), num_bytes_);
  data_ = nullptr;
  num_bytes_ = 0;
  size_ = 0;
}
} // namespace sxt::memmg
