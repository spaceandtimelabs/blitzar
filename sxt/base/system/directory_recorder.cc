/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/base/system/directory_recorder.h"

#include <cstdlib>
#include <filesystem>
#include <format>

#include "sxt/base/error/panic.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// try_get_directory
//--------------------------------------------------------------------------------------------------
static std::string_view try_get_directory() noexcept {
  static std::string res = [] {
    auto val = std::getenv("BLITZAR_DUMP_DIR");
    if (val != nullptr) {
      return std::string{val};
    } else {
      return std::string{};
    }
  }();
  return res;
}

//--------------------------------------------------------------------------------------------------
// counter
//--------------------------------------------------------------------------------------------------
std::atomic<unsigned> directory_recorder::counter_{0};

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
directory_recorder::directory_recorder(std::string base_name,
                                       std::string_view force_record_dir) noexcept {
  std::string_view dir;
  if (!force_record_dir.empty()) {
    dir = force_record_dir;
  } else {
    dir = try_get_directory();
  }
  if (dir.empty()) {
    return;
  }
  auto i = counter_++;
  name_ = std::format("{}/{}-{}", dir, base_name, i);
  try {
    std::filesystem::create_directory(name_);
  } catch (const std::exception& e) {
    baser::panic("failed to create directory {}: {}", name_, e.what());
  }
}
} // namespace sxt::bassy
