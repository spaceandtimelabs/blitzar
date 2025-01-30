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
#include "sxt/base/system/file_io.h"

#include <filesystem>

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// file_size
//--------------------------------------------------------------------------------------------------
size_t file_size(const char* filename) noexcept {
  try {
    return std::filesystem::file_size(filename);
  } catch (const std::exception& e) {
    baser::panic("failed to get file size of {}: {}", filename, e.what());
  }
}

//--------------------------------------------------------------------------------------------------
// write_file
//--------------------------------------------------------------------------------------------------
void write_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept {
  std::ofstream out{filename, std::ios::binary};
  if (!out.good()) {
    baser::panic("failed to open {}: {}", filename, std::strerror(errno));
  }
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  if (!out.good()) {
    baser::panic("failed to close {}: {}", filename, std::strerror(errno));
  }
}
} // namespace sxt::bassy
