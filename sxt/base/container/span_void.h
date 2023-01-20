#pragma once

#include <cstddef>

#include "sxt/base/error/assert.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// span_void_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class Derived, class T> class span_void_impl {
public:
  span_void_impl() noexcept = default;

  span_void_impl(T* data, size_t size, size_t element_size) noexcept
      : size_{size}, element_size_{element_size}, data_{data} {}

  T* data() const noexcept { return data_; }

  bool empty() const noexcept { return size_ == 0; }

  size_t size() const noexcept { return size_; }

  size_t element_size() const noexcept { return element_size_; }

  Derived subspan(size_t offset) const noexcept {
    SXT_DEBUG_ASSERT(offset <= size_);
    return {
        static_cast<void*>(static_cast<char*>(data_) + element_size_ * offset),
        size_ - offset,
        element_size_,
    };
  }

  Derived subspan(size_t offset, size_t size_p) const noexcept {
    SXT_DEBUG_ASSERT(offset <= size_);
    SXT_DEBUG_ASSERT(offset + size_p <= size_);
    return {
        static_cast<void*>(static_cast<char*>(data_) + element_size_ * offset),
        size_p,
        element_size_,
    };
  }

private:
  size_t size_ = 0;
  size_t element_size_ = 0;
  T* data_ = nullptr;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// span_cvoid
//--------------------------------------------------------------------------------------------------
class span_cvoid final : public detail::span_void_impl<span_cvoid, const void> {
public:
  using detail::span_void_impl<span_cvoid, const void>::span_void_impl;
};

//--------------------------------------------------------------------------------------------------
// span_void
//--------------------------------------------------------------------------------------------------
class span_void final : public detail::span_void_impl<span_void, void> {
public:
  using detail::span_void_impl<span_void, void>::span_void_impl;

  operator span_cvoid() const noexcept {
    return span_cvoid{this->data(), this->size(), this->element_size()};
  };
};
} // namespace sxt::basct
