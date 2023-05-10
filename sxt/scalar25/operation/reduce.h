/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// reduce33
//--------------------------------------------------------------------------------------------------
// Input:
//   s[0]+256*s[1]+...+256^31*s[31] = s
//   byte32
//
// Output:
//   s[0]+256*s[1]+...+256^31*s[31] = (t mod l)
//
// where t = s[0] + 256*s[1] + ... + 256^31*s[31] + 256^32*byte32
//
// where l = 2^252 + 27742317777372353535851937790883648493.
//
// Overwrites s in place.
CUDA_CALLABLE
void reduce33(s25t::element& s, uint8_t byte32) noexcept;

//--------------------------------------------------------------------------------------------------
// reduce32
//--------------------------------------------------------------------------------------------------
// Input:
//   s[0]+256*s[1]+...+256^31*s[31] = s
//
// Output:
//   s[0]+256*s[1]+...+256^31*s[31] = s mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493.
//
// Overwrites s in place.
CUDA_CALLABLE
inline void reduce32(s25t::element& s) noexcept { return reduce33(s, 0); }

//--------------------------------------------------------------------------------------------------
// reduce64
//--------------------------------------------------------------------------------------------------
// Input:
//   s[0]+256*s[1]+...+256^63*s[63] = s
//
// Output:
//   dest[0]+256*dest[1]+...+256^31*dest[31] = s mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493.
//
// Writes to dest.
CUDA_CALLABLE
void reduce64(s25t::element& dest, const uint8_t s[64]) noexcept;
} // namespace sxt::s25o
