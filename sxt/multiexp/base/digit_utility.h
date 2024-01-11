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

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void extract_digit(basct::span<uint8_t> digit, basct::cspan<uint8_t> e,
                                 size_t radix_log2, size_t digit_index) noexcept;

//--------------------------------------------------------------------------------------------------
// is_digit_zero
//--------------------------------------------------------------------------------------------------
bool is_digit_zero(basct::cspan<uint8_t> e, size_t radix_log2, size_t digit_index) noexcept;

//--------------------------------------------------------------------------------------------------
// get_last_digit
//--------------------------------------------------------------------------------------------------
size_t get_last_digit(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;
} // namespace sxt::mtxb
