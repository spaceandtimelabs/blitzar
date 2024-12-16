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

#include <format>
#include <string_view>

#include "sxt/base/log/log_impl.h"

namespace sxt::basl {
//--------------------------------------------------------------------------------------------------
// info
//--------------------------------------------------------------------------------------------------
inline void info(std::string_view s) noexcept { info_impl(s); }

template <class... Args> void info(std::format_string<Args...> fmt, Args&&... args) noexcept {
  info_impl(std::format(fmt, std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------------------
// error
//--------------------------------------------------------------------------------------------------
inline void error(std::string_view s) noexcept { error_impl(s); }

template <class... Args> void error(std::format_string<Args...> fmt, Args&&... args) noexcept {
  error_impl(std::format(fmt, std::forward<Args>(args)...));
}
} // namespace sxt::basl
