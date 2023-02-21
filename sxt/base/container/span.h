#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "sxt/base/container/span_void.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/macro/cuda_warning.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// span
//--------------------------------------------------------------------------------------------------
/**
 * Simple span class modeled after the c++20 class
 *
 * https://en.cppreference.com/w/cpp/container/span
 *
 * but 1) portable to earlier versions of c++ and 2) usable from device code
 */
template <class T> class span {
public:
  CUDA_CALLABLE
  span() noexcept : size_{0}, data_{nullptr} {}

  CUDA_CALLABLE
  span(T* first, size_t size) noexcept : size_{size}, data_{first} {}

  template <size_t N> CUDA_CALLABLE span(T (&arr)[N]) noexcept : size_{N}, data_{arr} {}

  CUDA_CALLABLE span(const span<std::remove_const_t<T>>& other) noexcept
    requires std::is_const_v<T>
      : size_{other.size()}, data_{other.data()} {}

  CUDA_DISABLE_HOSTDEV_WARNING
  template <class Cont>
  CUDA_CALLABLE span(Cont& cont) noexcept
    requires requires {
      { cont.data() } -> std::convertible_to<T*>;
      { cont.size() } -> std::convertible_to<size_t>;
    }
      : size_{cont.size()}, data_{cont.data()} {}

  CUDA_DISABLE_HOSTDEV_WARNING
  template <class Cont>
  CUDA_CALLABLE span(const Cont& cont) noexcept
    requires requires {
      { cont.data() } -> std::convertible_to<T*>;
      { cont.size() } -> std::convertible_to<size_t>;
    }
      : size_{cont.size()}, data_{cont.data()} {}

  operator span_cvoid() const noexcept {
    return {static_cast<const void*>(data_), size_, sizeof(T)};
  }

  operator span_void() const noexcept
    requires(!std::is_const_v<T>)
  {
    return {static_cast<void*>(data_), size_, sizeof(T)};
  }

  CUDA_CALLABLE
  T* data() const noexcept { return data_; }

  CUDA_CALLABLE
  size_t size() const noexcept { return size_; }

  CUDA_CALLABLE
  span subspan(size_t offset) const noexcept {
    assert(offset <= size_);
    return {
        data_ + offset,
        size_ - offset,
    };
  }

  CUDA_CALLABLE
  span subspan(size_t offset, size_t size_p) const noexcept {
    assert(offset + size_p <= size_);
    return {
        data_ + offset,
        size_p,
    };
  }

  CUDA_CALLABLE
  bool empty() const noexcept { return size_ == 0; }

  CUDA_CALLABLE
  T* begin() const noexcept { return data_; }
  T* end() const noexcept { return data_ + size_; }

  CUDA_CALLABLE
  T& operator[](size_t index) const noexcept {
    assert(index < size_);
    return data_[index];
  }

private:
  size_t size_;
  T* data_;
};

//--------------------------------------------------------------------------------------------------
// cspan
//--------------------------------------------------------------------------------------------------
template <class T> using cspan = span<const T>;
} // namespace sxt::basct
