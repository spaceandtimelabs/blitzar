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
#include "sxt/multiexp/pippenger2/window_width.h"

#include <charconv>
#include <cstdlib>
#include <cstring>

#include "sxt/base/error/panic.h"
#include "sxt/base/log/log.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// get_default_window_width_impl
//--------------------------------------------------------------------------------------------------
static unsigned get_default_window_width_impl() noexcept {
  auto s = std::getenv("BLITZAR_PARTITION_WINDOW_WIDTH");
  if (s == nullptr) {
    /* return 16; */
    return 14;
  }
  unsigned width;
  auto parse_result = std::from_chars(s, s + std::strlen(s), width);
  if (parse_result.ec != std::errc{}) {
    baser::panic("failed to parse partition window width {}", s);
  }
  if (width == 0) {
    baser::panic("partition window width cannot be zero");
  }
  return width;
}

//--------------------------------------------------------------------------------------------------
// get_default_window_width
//--------------------------------------------------------------------------------------------------
unsigned get_default_window_width() noexcept {
  static auto res = []() noexcept {
    auto res = get_default_window_width_impl();
    basl::info("using a default partition window width of {}", res);
    return res;
  }();
  return res;
}
} // namespace sxt::mtxpp2
