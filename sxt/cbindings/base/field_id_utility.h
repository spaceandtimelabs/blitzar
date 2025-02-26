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

#include <type_traits>

#include "sxt/base/error/panic.h"
#include "sxt/cbindings/base/field_id.h"
#include "sxt/scalar25/realization/field.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// switch_field_type
//--------------------------------------------------------------------------------------------------
template <class F> void switch_field_type(field_id_t id, F f) {
  switch (id) {
  case field_id_t::scalar25519:
    f(std::type_identity<s25t::element>{});
    break;
  default:
    baser::panic("unsupported field id {}", static_cast<unsigned>(id));
  }
}
} // namespace sxt::cbnb
