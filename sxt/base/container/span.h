#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "sxt/base/macro/cuda_callable.h"

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

  template <class Dummy = int, std::enable_if_t<std::is_const_v<T>, Dummy>* = nullptr>
  CUDA_CALLABLE span(const span<std::remove_const_t<T>>& other) noexcept
      : size_{other.size()}, data_{other.data()} {}

  template <class Cont,
            std::enable_if_t<
                std::is_convertible_v<decltype(std::declval<Cont&>().data()), T*> &&
                std::is_convertible_v<decltype(std::declval<Cont&>().size()), size_t>>* = nullptr>
  CUDA_CALLABLE span(Cont& cont) noexcept : size_{cont.size()}, data_{cont.data()} {}

  template <
      class Cont,
      std::enable_if_t<
          std::is_convertible_v<decltype(std::declval<const Cont&>().data()), T*> &&
          std::is_convertible_v<decltype(std::declval<const Cont&>().size()), size_t>>* = nullptr>
  CUDA_CALLABLE span(const Cont& cont) noexcept : size_{cont.size()}, data_{cont.data()} {}

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
    assert(offset <= size_);
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
  T& operator[](size_t index) const noexcept { return data_[index]; }

private:
  size_t size_;
  T* data_;
};

//--------------------------------------------------------------------------------------------------
// cspan
//--------------------------------------------------------------------------------------------------
template <class T> using cspan = span<const T>;
} // namespace sxt::basct
