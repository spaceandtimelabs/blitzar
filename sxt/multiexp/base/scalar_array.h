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

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// transpose_scalars_to_device
//--------------------------------------------------------------------------------------------------
/**
 * Copy scalars in host memory to the device and transpose their bytes. For example,
 * if there are two scalars, s1 and s2, of 4 bytes on the host layed out as follows
 *    s1[0], s1[1], s1[2], s1[3], s2[0], s2[1], s2[2], s2[3]
 *  then this function will copy the scalars to device memory and lay them out like this
 *    s1[0], s2[0], s1[1], s2[1], s1[2], s2[2], s1[3], s2[3]
 */
xena::future<> transpose_scalars_to_device(basct::span<uint8_t> array,
                                           basct::cspan<const uint8_t*> scalars,
                                           unsigned element_num_bytes, unsigned n) noexcept;
} // namespace sxt::mtxb
