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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::cgko {
//--------------------------------------------------------------------------------------------------
// mul_by_3b
//--------------------------------------------------------------------------------------------------
/**
 * For the Grumpkin curve, since b = -17, 3b = -51.
 * b3 is in Montgomery form.
 * See Algorithm 9 for details, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
inline void mul_by_3b(fgkt::element& h, const fgkt::element& p) noexcept {
  fgkt::element b3{0x985102072000010e, 0x66befc706194b935, 0x64a9867c966b3240, 0x9cabd298256ec00};

  fgko::mul(h, p, b3);
}
} // namespace sxt::cgko
