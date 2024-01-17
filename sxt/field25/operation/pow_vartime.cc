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
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/field25/operation/pow_vartime.h"

#include "sxt/field25/constant/one.h"
#include "sxt/field25/operation/mul.h"
#include "sxt/field25/operation/square.h"
#include "sxt/field25/type/element.h"

namespace sxt::f25o {
//--------------------------------------------------------------------------------------------------
// pow_vartime
//--------------------------------------------------------------------------------------------------
/**
 * Although this is labeled "vartime", it is only variable time with respect to the exponent.
 */
CUDA_CALLABLE
void pow_vartime(f25t::element& h, const f25t::element& f, const f25t::element& g) noexcept {
  f25t::element res = f25cn::one_v;

  for (int i = 3; i >= 0; --i) {
    f25t::element res_tmp;
    for (int j = 63; j >= 0; --j) {
      square(res_tmp, res);
      res = res_tmp;
      if (((g[i] >> j) & 1) == 1) {
        mul(res_tmp, f, res);
        res = res_tmp;
      }
    }
  }

  h = res;
}
} // namespace sxt::f25o
