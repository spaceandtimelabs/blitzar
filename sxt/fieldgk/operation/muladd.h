/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"

namespace sxt::fgko {
//--------------------------------------------------------------------------------------------------
// muladd
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void muladd(fgkt::element& s, const fgkt::element& a, const fgkt::element& b,
                                 const fgkt::element& c) noexcept {
  auto cp = c;
  mul(s, a, b);
  add(s, s, cp);
}
} // namespace sxt::fgko
