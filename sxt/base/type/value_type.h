/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <type_traits>

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// value_type
//--------------------------------------------------------------------------------------------------
/**
 * Mirror https://en.cppreference.com/w/cpp/experimental/ranges/iterator/value_type
 */
template <class T> struct value_type {};

template <class T> struct value_type<T*> {
  using type = std::remove_cv_t<T>;
};

template <class T>
  requires std::is_array_v<T>
struct value_type<T> : value_type<std::decay_t<T>> {};

template <class T> struct value_type<const T> : value_type<std::decay_t<T>> {};

template <class T>
  requires requires { typename T::value_type; }
struct value_type<T> {
  using type = typename T::value_type;
};

template <class T>
  requires requires { typename T::element_type; }
struct value_type<T> {
  using type = typename T::element_type;
};

//--------------------------------------------------------------------------------------------------
// value_type_t
//--------------------------------------------------------------------------------------------------
template <class T> using value_type_t = typename value_type<T>::type;
} // namespace sxt::bast
