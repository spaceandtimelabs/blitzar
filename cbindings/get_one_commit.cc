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
#include "cbindings/get_one_commit.h"

#include <iostream>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_curve25519_get_one_commit
//--------------------------------------------------------------------------------------------------
int sxt_curve25519_get_one_commit(struct sxt_ristretto255* one_commit, uint64_t n) {
  SXT_RELEASE_ASSERT(
      sxt::cbn::is_backend_initialized(),
      "backend uninitialized in the `sxt_curve25519_get_one_commit` c binding function");
  SXT_RELEASE_ASSERT(
      one_commit != nullptr,
      "one_commit input to `sxt_curve25519_get_one_commit` c binding function is null");

  reinterpret_cast<sxt::c21t::element_p3*>(one_commit)[0] =
      sxt::sqcgn::get_precomputed_one_commit(n);

  return 0;
}
