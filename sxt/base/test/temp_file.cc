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
#include "sxt/base/test/temp_file.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sxt/base/io/print.h"

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
temp_file::temp_file(std::ios_base::openmode openmode) noexcept
    : name_{std::tmpnam(nullptr)}, out_{name_, openmode} {}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
temp_file::~temp_file() noexcept {
  out_.close();
  auto rcode = std::remove(name_.c_str());
  if (rcode != 0) {
    basio::println(stderr, "failed to close file {}: {}", name_, std::strerror(errno));
    std::abort();
  }
}
} // namespace sxt::bastst
