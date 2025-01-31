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
#pragma once

#include <atomic>
#include <string>
#include <string_view>

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// directory_recorder
//--------------------------------------------------------------------------------------------------
class directory_recorder {
public:
  directory_recorder(std::string base_name, std::string_view force_record_dir = {}) noexcept;

  bool recording() const noexcept { return !name_.empty(); }

  std::string_view dir() const noexcept { return name_; }

private:
  static std::atomic<unsigned> counter_;
  std::string name_;
};
} // namespace sxt::bassy
