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
#include "sxt/execution/async/synchronization.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// await_stream
//--------------------------------------------------------------------------------------------------
future<> await_stream(bast::raw_stream_t stream) noexcept {
  promise<> p;
  future<> res{p};
  basdv::event event;
  basdv::record_event(event, stream);
  xens::get_scheduler().schedule(
      std::make_unique<gpu_computation_event<>>(std::move(event), std::move(p)));
  return res;
}

//--------------------------------------------------------------------------------------------------
// await_and_own_stream
//--------------------------------------------------------------------------------------------------
future<> await_and_own_stream(basdv::stream&& stream) noexcept {
  promise<> p;
  future<> res{p};
  basdv::event event;
  basdv::record_event(event, stream);
  computation_handle handle;
  handle.add_stream(std::move(stream));
  xens::get_scheduler().schedule(
      std::make_unique<gpu_computation_event<>>(std::move(event), std::move(handle), std::move(p)));
  return res;
}
} // namespace sxt::xena
