#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "sxt/base/container/span_void.h"
#include "sxt/base/type/polymorphic_allocator.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::memmg {
//--------------------------------------------------------------------------------------------------
// managed_array
//--------------------------------------------------------------------------------------------------
// void
template <> class managed_array<void> {
public:
  using allocator_type = bast::polymorphic_allocator;

  // constructor
  managed_array() noexcept = default;

  managed_array(allocator_type alloc) noexcept : alloc_{alloc} {}

  managed_array(void* data, size_t size, size_t num_bytes, allocator_type alloc = {}) noexcept;

  managed_array(const managed_array& other) noexcept;

  managed_array(const managed_array& other, allocator_type alloc) noexcept;

  managed_array(managed_array&& other) noexcept;

  managed_array(managed_array&& other, allocator_type alloc) noexcept;

  // assignment
  managed_array& operator=(const managed_array& other) noexcept;
  managed_array& operator=(managed_array&& other) noexcept;

  // destructor
  ~managed_array() noexcept;

  // accessors
  allocator_type get_allocator() const noexcept { return alloc_; }

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }

  size_t size() const noexcept { return size_; }

  size_t num_bytes() const noexcept { return num_bytes_; }

  bool empty() const noexcept { return size_ == 0; }

  // methods
  void reset() noexcept;

private:
  allocator_type alloc_;
  void* data_{nullptr};
  size_t size_{0};
  size_t num_bytes_{0};
};

// general
template <class T> class managed_array {
  static_assert(std::is_trivially_destructible_v<T>);

public:
  using allocator_type = bast::polymorphic_allocator;

  // constructor
  managed_array() noexcept = default;

  managed_array(allocator_type alloc) noexcept : data_{alloc} {}

  explicit managed_array(size_t size, allocator_type alloc = {}) noexcept
      : data_{static_cast<void*>(alloc.allocate(size * sizeof(T))), size, size * sizeof(T), alloc} {
  }

  managed_array(T* data, size_t size, allocator_type alloc = {}) noexcept
      : data_{static_cast<void*>(data), size, size * sizeof(T), alloc} {}

  managed_array(std::initializer_list<T> values, allocator_type alloc = {})
      : data_{reinterpret_cast<void*>(alloc.allocate(values.size() * sizeof(T))), values.size(),
              values.size() * sizeof(T), alloc} {
    std::copy(values.begin(), values.end(), this->data());
  }

  managed_array(const managed_array& other) noexcept = default;

  managed_array(const managed_array& other, allocator_type alloc) noexcept
      : data_{other.data_, alloc} {}

  managed_array(managed_array&& other) noexcept = default;

  managed_array(managed_array&& other, allocator_type alloc) noexcept
      : data_{std::move(other.data_), alloc} {}

  managed_array& operator=(const managed_array&) noexcept = default;
  managed_array& operator=(managed_array&&) noexcept = default;

  // conversion
  operator managed_array<void>&() & noexcept { return data_; }

  operator managed_array<void>&&() && noexcept { return std::move(data_); }

  operator basct::span_void() noexcept {
    return basct::span_void{data_.data(), data_.size(), sizeof(T)};
  }

  // accessors
  allocator_type get_allocator() const noexcept { return data_.get_allocator(); }

  T* data() noexcept { return static_cast<T*>(data_.data()); }
  const T* data() const noexcept { return static_cast<const T*>(data_.data()); }

  size_t size() const noexcept { return data_.size(); }

  size_t num_bytes() const noexcept { return data_.num_bytes(); }

  bool empty() const noexcept { return data_.empty(); }

  T* begin() noexcept { return this->data(); }
  T* end() noexcept { return this->data() + data_.size(); }
  const T* begin() const noexcept { return this->data(); }
  const T* end() const noexcept { return this->data() + data_.size(); }

  // methods
  void reset() noexcept { data_.reset(); }

  // operator[]
  T& operator[](size_t index) noexcept { return this->data()[index]; }

  const T& operator[](size_t index) const noexcept { return this->data()[index]; }

private:
  managed_array<void> data_;
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
template <class T1, class T2>
auto operator==(const managed_array<T1>& lhs, const managed_array<T2>& rhs) noexcept
    -> decltype(*lhs.data() == *rhs.data()) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
template <class T1, class T2>
auto operator!=(const managed_array<T1>& lhs, const managed_array<T2>& rhs) noexcept
    -> decltype(*lhs.data() == *rhs.data()) {
  return !(lhs == rhs);
}
} // namespace sxt::memmg
