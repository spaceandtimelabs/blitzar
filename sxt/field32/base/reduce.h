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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/int.h"

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
/*
 * Given unreduced coefficients `z[0], ..., z[9]` of any size,
 * carry and reduce them mod p to obtain a `FieldElement2625`
 * whose coefficients have excess `b < 0.007`.
 *
 * In other words, each coefficient of the result is bounded by
 * either `2^(25 + 0.007)` or `2^(26 + 0.007)`, as appropriate.
 *
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/field.rs#L329-L390
 */
CUDA_CALLABLE
void reduce(uint32_t h[10], const uint64_t f[10]) noexcept;
} // namespace sxt::f32b
