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

#include "sxt/base/container/span.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// proof_descriptor
//--------------------------------------------------------------------------------------------------
/**
 * Description of inputs used for an inner product proof.
 *
 * In our version of an inner product proof, the prover is establishing that
 *    <a_vector, b_vector> = product
 * where
 *      commit_a = sum_i a[i] * g[i]
 * and b_vector is known to the verifier.
 */
struct proof_descriptor {
  basct::cspan<s25t::element> b_vector;
  basct::cspan<c32t::element_p3> g_vector;
  const c32t::element_p3* q_value = nullptr;
};
} // namespace sxt::prfip
