#pragma once

#include <iterator>
#include <type_traits>

#include "sxt/base/type/remove_cvref.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// has_distance_to_v
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct has_distance_to {
  static constexpr bool value = false;
};

template <class T>
struct has_distance_to<T, std::void_t<decltype(std::declval<T>().distance_to(std::declval<T>()))>> {
  static constexpr bool value = true;
};

template <class T> static constexpr bool has_distance_to_v = has_distance_to<T>::value;
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_difference_type
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct iterator_difference_type {
  using type = std::ptrdiff_t;
};

template <class T> struct iterator_difference_type<T, std::enable_if_t<has_distance_to_v<T>>> {
  using type = decltype(std::declval<T>().distance_to(std::declval<T>()));
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_value_type
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct iterator_value_type {
  using type = bast::remove_cvref_t<decltype(*std::declval<T>())>;
};

template <class T>
struct iterator_value_type<
    T, std::enable_if_t<std::is_same_v<typename T::value_type, typename T::value_type>>> {
  using type = typename T::value_type;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_reference
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct iterator_reference {
  using type = decltype(std::declval<T>().dereference());
};

template <class T>
struct iterator_reference<
    T, std::enable_if_t<std::is_same_v<typename T::reference, typename T::reference>>> {
  using type = typename T::reference;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_pointer
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct iterator_pointer {
  using type = std::remove_reference_t<decltype(std::declval<T>().dereference())>*;
};

template <class T>
struct iterator_pointer<
    T, std::enable_if_t<std::is_same_v<typename T::pointer, typename T::pointer>>> {
  using type = typename T::pointer;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// has_increment
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct has_increment {
  static constexpr bool value = false;
};

template <class T> struct has_increment<T, std::void_t<decltype(std::declval<T>().increment())>> {
  static constexpr bool value = true;
};

template <class T> static constexpr bool has_increment_v = has_increment<T>::value;
} // namespace detail

//--------------------------------------------------------------------------------------------------
// has_decrement
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct has_decrement {
  static constexpr bool value = false;
};

template <class T> struct has_decrement<T, std::void_t<decltype(std::declval<T>().decrement())>> {
  static constexpr bool value = true;
};

template <class T> static constexpr bool has_decrement_v = has_decrement<T>::value;
} // namespace detail

//--------------------------------------------------------------------------------------------------
// has_advance
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct has_advance {
  static constexpr bool value = false;
};

template <class T>
struct has_advance<T, std::void_t<decltype(std::declval<T>().advance(std::ptrdiff_t{}))>> {
  static constexpr bool value = true;
};

template <class T> static constexpr bool has_advance_v = has_advance<T>::value;
} // namespace detail

//--------------------------------------------------------------------------------------------------
// is_valid_advance
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class Delta, class = void> struct is_valid_advance {
  static constexpr bool value = false;
};

template <class T, class Delta>
struct is_valid_advance<T, Delta, std::void_t<decltype(std::declval<T>().advance(Delta{}))>> {
  static constexpr bool value = true;
};

template <class T, class Delta>
static constexpr bool is_valid_advance_v = is_valid_advance<T, Delta>::value;
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_category
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class = void> struct iterator_category {
  using type =
      std::conditional_t<has_advance_v<T>, std::random_access_iterator_tag,
                         std::conditional_t<has_decrement_v<T>, std::bidirectional_iterator_tag,
                                            std::forward_iterator_tag>>;
};

template <class T>
struct iterator_category<T, std::enable_if_t<std::is_same_v<typename T::iterator_category,
                                                            typename T::iterator_category>>> {
  using type = typename T::iterator_category;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// arrow_proxy
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class Reference> struct arrow_proxy {
  Reference reference;
  Reference* operator->() noexcept { return &reference; }
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// iterator_facade
//--------------------------------------------------------------------------------------------------
template <class Derived> class iterator_facade {
public:
  decltype(auto) operator*() const noexcept { return this->self().dereference(); }

  auto operator->() const {
    decltype(auto) reference = **this;
    if constexpr (std::is_reference_v<decltype(reference)>) {
      return std::addressof(reference);
    } else {
      return detail::arrow_proxy<decltype(reference)>{std::move(reference)};
    }
  }

  // operator++
  template <class T = Derived, std::enable_if_t<detail::has_increment_v<T>>* = nullptr>
  Derived& operator++() noexcept {
    this->self().increment();
    return self();
  }

  template <class T = Derived,
            std::enable_if_t<!detail::has_increment_v<T> && detail::has_advance_v<T>>* = nullptr>
  Derived& operator++() noexcept {
    this->self().advance(1);
    return self();
  }

  Derived operator++(int) noexcept {
    auto result = this->self();
    ++*this;
    return result;
  }

  // operator--
  template <class T = Derived, std::enable_if_t<detail::has_decrement_v<T>>* = nullptr>
  Derived& operator--() noexcept {
    this->self().decrement();
    return self();
  }

  template <class T = Derived,
            std::enable_if_t<!detail::has_decrement_v<T> && detail::has_advance_v<T>>* = nullptr>
  Derived& operator--() noexcept {
    this->self().advance(-1);
    return self();
  }

  Derived operator--(int) noexcept {
    auto result = this->self();
    --*this;
    return result;
  }

  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  decltype(auto) operator[](Delta delta) const noexcept {
    return *(this->self() + delta);
  }

private:
  Derived& self() noexcept { return static_cast<Derived&>(*this); }

  const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }

  // operator+=
  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived& operator+=(Derived& self, Delta delta) noexcept {
    self.advance(delta);
    return self;
  }

  // operator-=
  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived& operator-=(Derived& self, Delta delta) noexcept {
    self.advance(-delta);
    return self;
  }

  // operator+
  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived operator+(Derived lhs, Delta delta) noexcept {
    lhs += delta;
    return lhs;
  }

  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived operator+(Delta delta, Derived rhs) noexcept {
    rhs += delta;
    return rhs;
  }

  // operator-
  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived operator-(Derived lhs, Delta delta) noexcept {
    lhs -= delta;
    return lhs;
  }

  template <class Delta, std::enable_if_t<detail::is_valid_advance_v<Derived, Delta>>* = nullptr>
  friend Derived operator-(Delta delta, Derived rhs) noexcept {
    rhs -= delta;
    return rhs;
  }

  template <class T = Derived, std::enable_if_t<detail::has_distance_to_v<T>>* = nullptr>
  friend auto operator-(const Derived& lhs, const Derived& rhs) noexcept {
    return rhs.distance_to(lhs);
  }

  // operator==
  friend bool operator==(const Derived& lhs, const Derived& rhs) noexcept {
    return lhs.equal_to(rhs);
  }

  // operator!=
  friend bool operator!=(const Derived& lhs, const Derived& rhs) noexcept {
    return !lhs.equal_to(rhs);
  }

  // operator<
  friend bool operator<(const Derived& lhs, const Derived& rhs) noexcept { return lhs - rhs < 0; }

  // operator>
  friend bool operator>(const Derived& lhs, const Derived& rhs) noexcept { return lhs - rhs > 0; }

  // operator<=
  friend bool operator<=(const Derived& lhs, const Derived& rhs) noexcept { return lhs - rhs <= 0; }

  // operator>=
  friend bool operator>=(const Derived& lhs, const Derived& rhs) noexcept { return lhs - rhs >= 0; }
};

//--------------------------------------------------------------------------------------------------
// iterator_traits_impl
//--------------------------------------------------------------------------------------------------
template <class T> struct iterator_traits_impl {
  using difference_type = typename detail::iterator_difference_type<T>::type;
  using value_type = typename detail::iterator_value_type<T>::type;
  using reference = typename detail::iterator_reference<T>::type;
  using pointer = typename detail::iterator_pointer<T>::type;
  using iterator_category = typename detail::iterator_category<T>::type;
};
} // namespace sxt::basit
