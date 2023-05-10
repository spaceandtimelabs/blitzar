/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
struct element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// workspace
//--------------------------------------------------------------------------------------------------
/**
 * Proving an inner product proceeds over multiple rounds. This abstraction allows a backend for the
 * computational steps to persist data between prover rounds.
 */
class workspace {
public:
  virtual ~workspace() noexcept = default;

  /**
   * On the final round of an inner product proof for the product of vectors <a, b> where b is
   * known, the prover will have repeatedly folded the vector a down to a single element a'. This
   * function provides an accessor to the a' value.
   */
  virtual xena::future<void> ap_value(s25t::element& value) const noexcept = 0;
};
} // namespace sxt::prfip
