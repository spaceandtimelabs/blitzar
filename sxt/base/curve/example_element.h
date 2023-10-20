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

#include <cstdint>
#include <iostream>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::bascrv {
//--------------------------------------------------------------------------------------------------
// element97
//--------------------------------------------------------------------------------------------------
<<<<<<< HEAD
=======
/**
 * Integer arithmetic mod 97.
 *
 * Provides a minimal implementation of the curve element concent that can be used for writing
 * tests.
 */
>>>>>>> 717fe89b07c804f114843c360e19535d9690f629
struct element97 {
  uint32_t value;
  bool marked = false;

  element97() noexcept = default;

  constexpr element97(uint32_t val) noexcept : value{val % 97u} {}

<<<<<<< HEAD
  static constexpr element97 identity() noexcept {
    return {0};
  }
=======
  static constexpr element97 identity() noexcept { return {0}; }
>>>>>>> 717fe89b07c804f114843c360e19535d9690f629

  bool operator==(const element97&) const noexcept = default;
};

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& out, const element97& e) noexcept {
  out << e.value;
  return out;
}

//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void double_element(element97& res, const element97& e) noexcept {
  res.value = (e.value + e.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void neg(element97& res, const element97& e) noexcept {
  res.value = (97u - e.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void cneg(element97& res, int b) noexcept {
  if (b) {
    res.value = (97u - res.value) % 97u;
  }
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void add(element97& res, const element97& x, const element97& y) noexcept {
  res.value = (x.value + y.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void add_inplace(element97& res, element97& x) noexcept {
  res.value = (res.value + x.value) % 97u;
<<<<<<< HEAD
  x = 13; // simulate a destructive add by setting x to an arbitrary value 
=======
  x = 13; // simulate a destructive add by setting x to an arbitrary value
>>>>>>> 717fe89b07c804f114843c360e19535d9690f629
}

//--------------------------------------------------------------------------------------------------
// mark
//--------------------------------------------------------------------------------------------------
<<<<<<< HEAD
inline CUDA_CALLABLE void mark(element97& res) noexcept {
  res.marked = true;
}
=======
inline CUDA_CALLABLE void mark(element97& res) noexcept { res.marked = true; }
>>>>>>> 717fe89b07c804f114843c360e19535d9690f629

//--------------------------------------------------------------------------------------------------
// is_marked
//--------------------------------------------------------------------------------------------------
<<<<<<< HEAD
inline CUDA_CALLABLE bool is_marked(const element97& e) noexcept {
  return e.marked;
}
=======
inline CUDA_CALLABLE bool is_marked(const element97& e) noexcept { return e.marked; }
>>>>>>> 717fe89b07c804f114843c360e19535d9690f629
} // namespace sxt::bascrv
