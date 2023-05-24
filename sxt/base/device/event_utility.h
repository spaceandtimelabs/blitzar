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

#include "sxt/base/type/raw_cuda_event.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// record_event
//--------------------------------------------------------------------------------------------------
void record_event(bast::raw_cuda_event_t event, bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_wait_on_event
//--------------------------------------------------------------------------------------------------
void async_wait_on_event(bast::raw_stream_t stream, bast::raw_cuda_event_t event) noexcept;
} // namespace sxt::basdv
