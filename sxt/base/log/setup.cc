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
#include "sxt/base/log/setup.h"

#include <cctype>
#include <cstdlib>

#include "spdlog/spdlog.h"

namespace sxt::basl {
//--------------------------------------------------------------------------------------------------
// set_log_level
//--------------------------------------------------------------------------------------------------
static void set_log_level() noexcept {
  auto level = std::getenv("BLITZAR_LOG_LEVEL");
  if (level == nullptr) {
    spdlog::set_level(spdlog::level::err);
    return;
  }
  std::string s{level};
  for (auto& c : s) {
    c = std::tolower(c);
  }
  if (s == "error") {
    spdlog::set_level(spdlog::level::debug);
  } else if (s == "debug") {
    spdlog::set_level(spdlog::level::debug);
  } else if (s == "warn") {
    spdlog::set_level(spdlog::level::warn);
  } else if (s == "info") {
    spdlog::set_level(spdlog::level::info);
  } else if (s == "trace") {
    spdlog::set_level(spdlog::level::trace);
  } else if (s == "critical") {
    spdlog::set_level(spdlog::level::critical);
  } else if (s == "off") {
    spdlog::set_level(spdlog::level::off);
  }
}

//--------------------------------------------------------------------------------------------------
// setup_logger
//--------------------------------------------------------------------------------------------------
void setup_logger() noexcept {
  static thread_local bool is_initialized = [] {
    set_log_level();
    return true;
  }();
  (void)is_initialized;
}
} // namespace sxt::basl
