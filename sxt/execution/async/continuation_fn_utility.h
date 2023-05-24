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

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// continuation_fn_result_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class F, class T> struct continuation_fn_result_impl {
  using type = std::invoke_result_t<F&, T&&>;
};

template <class F> struct continuation_fn_result_impl<F, void> {
  using type = std::invoke_result_t<F&>;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// continuation_fn_result_t
//--------------------------------------------------------------------------------------------------
template <class F, class T>
using continuation_fn_result_t = typename detail::continuation_fn_result_impl<F, T>::type;
} // namespace sxt::xena
