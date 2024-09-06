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

#include <concepts>
#include <format>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

#include "sxt/base/error/stacktrace.h"

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// panic_message
//--------------------------------------------------------------------------------------------------
struct panic_message {
  template <class T>
    requires std::constructible_from<std::string_view, T>
  panic_message(const T& s, std::source_location loc = std::source_location::current()) noexcept
      : s{s}, loc{loc} {}

  std::string_view s;
  std::source_location loc;
};

//--------------------------------------------------------------------------------------------------
// panic_format
//--------------------------------------------------------------------------------------------------
template <class... Args> struct panic_format {
  template <class T>
    requires std::constructible_from<std::format_string<Args...>, T>
  consteval panic_format(const T& s,
                         std::source_location loc = std::source_location::current()) noexcept
      : fmt{s}, loc{loc} {}

  std::format_string<Args...> fmt;
  std::source_location loc;
};

//--------------------------------------------------------------------------------------------------
// panic_with_message
//--------------------------------------------------------------------------------------------------
[[noreturn]] void panic_with_message(std::string_view file, int line, std::string_view msg,
                                     const std::string& trace = stacktrace()) noexcept;

//--------------------------------------------------------------------------------------------------
// panic
//--------------------------------------------------------------------------------------------------
/**
 * Adopted from https://buildingblock.ai/panic
 */
[[noreturn]] inline void panic(panic_message msg) noexcept {
  panic_with_message(msg.loc.file_name(), msg.loc.line(), msg.s);
}

#if 0
template <class... Args>
[[noreturn]] void panic(panic_format<std::type_identity_t<Args>...> fmt, Args&&... args) noexcept
  requires(sizeof...(Args) > 0)
{
  panic_with_message(fmt.loc.file_name(), fmt.loc.line(),
                     std::format(fmt.fmt, std::forward<Args>(args)...));
}
#endif

template <class... Args>
[[noreturn]] void panic(panic_message fmt, Args&&... args) noexcept
  requires(sizeof...(Args) > 0)
{
  panic_with_message(fmt.loc.file_name(), fmt.loc.line(),
                     std::vformat(fmt.s, std::make_format_args(args...)));
}
} // namespace sxt::baser
