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

#include <string_view>

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// panic
//--------------------------------------------------------------------------------------------------
/**
 * Note: This technique for getting the file and line is technically not standard compliant, but
 * it works with most compilers (gcc, clang) and the standard-friendly approach requires c++20
 * and support is flaky:
 * https://en.cppreference.com/w/cpp/utility/source_location
 */
[[noreturn]] void panic(std::string_view message, int line = __builtin_LINE(),
                        const char* file = __builtin_FILE()) noexcept;
} // namespace sxt::baser
