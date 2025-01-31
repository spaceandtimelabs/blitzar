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

#include <cerrno>
#include <cstddef>
#include <fstream>

#include "sxt/base/container/span.h"
#include "sxt/base/error/panic.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// file_size
//--------------------------------------------------------------------------------------------------
size_t file_size(const char* filename) noexcept;

//--------------------------------------------------------------------------------------------------
// write_file
//--------------------------------------------------------------------------------------------------
void write_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept;

template <class T> void write_file(const char* filename, basct::cspan<T> values) noexcept {
  basct::cspan<uint8_t> bytes{reinterpret_cast<const uint8_t*>(values.data()),
                              values.size() * sizeof(T)};
  write_file(filename, bytes);
}

template <class T> void write_file(std::string_view filename, basct::cspan<T> values) noexcept {
  write_file(std::string{filename}.c_str(), values);
}

//--------------------------------------------------------------------------------------------------
// read_file
//--------------------------------------------------------------------------------------------------
template <class T> void read_file(std::vector<T>& values, const char* filename) noexcept {
  auto sz = file_size(filename);
  if (sz % sizeof(T) != 0) {
    baser::panic("{} file size {} is not a multiple of element size {}", filename, sz, sizeof(T));
  }
  auto n = sz / sizeof(T);
  std::ifstream in{filename, std::ios::binary};
  if (!in.good()) {
    baser::panic("failed to open {}: {}", filename, std::strerror(errno));
  }
  values.resize(n);
  in.read(reinterpret_cast<char*>(values.data()), sz);
  if (!in.good()) {
    baser::panic("failed to read {}: {}", filename, std::strerror(errno));
  }
}

template <class T> void read_file(std::vector<T>& values, std::string_view filename) noexcept {
  read_file(values, std::string{filename}.c_str());
}
} // namespace sxt::bassy
