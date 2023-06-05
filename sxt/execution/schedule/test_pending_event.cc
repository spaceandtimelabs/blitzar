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
#include "sxt/execution/schedule/test_pending_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
test_pending_event::test_pending_event(int id, std::function<void(int, int)> f) noexcept
    : id_{id}, f_{std::move(f)} {}

//--------------------------------------------------------------------------------------------------
// invoke
//--------------------------------------------------------------------------------------------------
void test_pending_event::invoke(int device) noexcept {
  if (f_) {
    f_(id_, device);
  }
}
} // namespace sxt::xens
