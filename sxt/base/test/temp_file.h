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

#include <fstream>
#include <string>

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// temp_file
//--------------------------------------------------------------------------------------------------
/**
 * Set up a temporary file that is deleted in the destructor.
 *
 * This is meant to make easier to write tests involving files.
 */
class temp_file {
public:
  explicit temp_file(std::ios_base::openmode = std::ios_base::out) noexcept;

  ~temp_file() noexcept;

  const std::string& name() const noexcept { return name_; }

  std::ofstream& stream() noexcept { return out_; }

private:
  std::string name_;
  std::ofstream out_;
};
} // namespace sxt::bastst
