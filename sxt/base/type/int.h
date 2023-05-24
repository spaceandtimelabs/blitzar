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

#include <cstddef>
#include <cstdint>

using uint128_t = __uint128_t;

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// sized_int_t_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <size_t> struct sized_int_t_impl {};

template <> struct sized_int_t_impl<8> {
  using type = int8_t;
};

template <> struct sized_int_t_impl<16> {
  using type = int16_t;
};

template <> struct sized_int_t_impl<32> {
  using type = int32_t;
};

template <> struct sized_int_t_impl<64> {
  using type = int64_t;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// sized_int_t
//--------------------------------------------------------------------------------------------------
template <size_t K> using sized_int_t = typename detail::sized_int_t_impl<K>::type;
} // namespace sxt::bast
