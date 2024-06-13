/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include <type_traits>

#include "sxt/base/error/panic.h"
#include "sxt/cbindings/base/curve_id.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve_bng1/operation/add.h"
#include "sxt/curve_bng1/operation/double.h"
#include "sxt/curve_bng1/operation/neg.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/double.h"
#include "sxt/curve_g1/operation/neg.h"
#include "sxt/curve_g1/type/element_p2.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// switch_curve_type
//--------------------------------------------------------------------------------------------------
template <class F> void switch_curve_type(curve_id_t id, F f) {
  switch (id) {
  case curve_id_t::curve25519:
    f(std::type_identity<c21t::compact_element>{}, std::type_identity<c21t::element_p3>{});
    break;
  case curve_id_t::bls12_381:
    f(std::type_identity<cg1t::compact_element>{}, std::type_identity<cg1t::element_p2>{});
    break;
  case curve_id_t::bn254:
    f(std::type_identity<cn1t::compact_element>{}, std::type_identity<cn1t::element_p2>{});
    break;
  default:
    baser::panic("unsupported curve id {}", static_cast<unsigned>(id));
  }
}
} // namespace sxt::cbnb
