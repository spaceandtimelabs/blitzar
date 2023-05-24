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
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
struct element;
}

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// scalar_multiply255
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]

 Preconditions:
 a[31] <= 127

 p is public
 */
CUDA_CALLABLE
void scalar_multiply255(c21t::element_p3& h, const unsigned char* a,
                        const c21t::element_p3& p) noexcept;

//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(c21t::element_p3& h, basct::cspan<uint8_t> a,
                     const c21t::element_p3& p) noexcept;

template <class T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>* = nullptr>
void scalar_multiply(c21t::element_p3& h, T a, const c21t::element_p3& p) noexcept {
  scalar_multiply(h, basct::cspan<uint8_t>{reinterpret_cast<uint8_t*>(&a), sizeof(a)}, p);
}

void scalar_multiply(c21t::element_p3& h, const s25t::element& a,
                     const c21t::element_p3& p) noexcept;
} // namespace sxt::c21o
