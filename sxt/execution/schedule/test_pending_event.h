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

#include <functional>

#include "sxt/execution/schedule/pending_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// test_pending_event
//--------------------------------------------------------------------------------------------------
class test_pending_event final : public pending_event {
public:
  test_pending_event(int id, std::function<void(int, int)> f = {}) noexcept;

  // pending_event
  void invoke(int device) noexcept override;

private:
  int id_;
  std::function<void(int, int)> f_;
};
} // namespace sxt::xens
