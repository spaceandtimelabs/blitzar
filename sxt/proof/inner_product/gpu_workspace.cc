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
#include "sxt/proof/inner_product/gpu_workspace.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
gpu_workspace::gpu_workspace() noexcept
    : alloc{memr::get_pinned_resource()}, a_vector{memr::get_device_resource()},
      b_vector{memr::get_device_resource()}, g_vector{memr::get_device_resource()} {}

//--------------------------------------------------------------------------------------------------
// ap_value
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_workspace::ap_value(s25t::element& value) const noexcept {
  if (use_new) {
    value = this->a_vectorX[0];
    co_return;
  } else {
    SXT_DEBUG_ASSERT(this->a_vector.size() == 1);
    memmg::managed_array<s25t::element> value_p{1, memr::get_pinned_resource()};
    basdv::stream stream;
    basdv::async_copy_device_to_host(value_p, this->a_vector, stream);
    co_await xendv::await_and_own_stream(std::move(stream));
    value = value_p[0];
  }
}
} // namespace sxt::prfip
