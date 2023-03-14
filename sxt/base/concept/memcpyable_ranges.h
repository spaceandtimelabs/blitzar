#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

namespace sxt::bascpt {
//--------------------------------------------------------------------------------------------------
// memcpyable_ranges
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src, class IterDst = decltype(std::begin(std::declval<Dst>())),
          class IterSrc = decltype(std::begin(std::declval<const Src&>())),
          class T = typename std::iterator_traits<IterDst>::value_type,
          class Tp = typename std::iterator_traits<IterSrc>::value_type>
concept memcpyable_ranges =
    std::same_as<T, Tp> && std::contiguous_iterator<IterDst> && std::contiguous_iterator<IterSrc> &&
    std::output_iterator<IterDst, T> && requires(Dst dst, const Src& src) {
      // clang-format off
      { std::size(dst) } -> std::convertible_to<size_t>;
      { std::size(src) } -> std::convertible_to<size_t>;
      // clang-format on
    };
} // namespace sxt::bascpt
