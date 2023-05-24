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

#include "cbindings/proofs_gpu_api.h"
#include "sxt/cbindings/backend/computational_backend.h"

namespace sxt::cbn {

//--------------------------------------------------------------------------------------------------
// is_backend_initialized
//--------------------------------------------------------------------------------------------------
bool is_backend_initialized() noexcept;

//--------------------------------------------------------------------------------------------------
// get_backend
//--------------------------------------------------------------------------------------------------
cbnbck::computational_backend* get_backend() noexcept;

//--------------------------------------------------------------------------------------------------
// reset_backend_for_testing
//--------------------------------------------------------------------------------------------------
void reset_backend_for_testing() noexcept;

} // namespace sxt::cbn
