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
/*
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_g1/operation/sub.h"

#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/neg.h"
#include "sxt/curve_g1/type/element_p2.h"

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sub(cg1t::element_p2& r, const cg1t::element_p2& p, const cg1t::element_p2& q) noexcept {
  cg1t::element_p2 neg_q;
  neg(neg_q, q);
  add(r, p, neg_q);
}
} // namespace sxt::cg1o
