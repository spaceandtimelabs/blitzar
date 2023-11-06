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
#include "sxt/multiexp/bucket_method/accumulation.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_exponents_viewable
//--------------------------------------------------------------------------------------------------
basct::cspan<const uint8_t>
make_exponents_viewable(memmg::managed_array<uint8_t>& exponents_viewable_data,
                        basct::cspan<const uint8_t*> exponents, const basit::index_range& rng,
                        const basdv::stream& stream) noexcept {
  static constexpr size_t exponent_size = 32; // hard coded for now
  auto num_outputs = exponents.size();
  auto n = rng.size();
  exponents_viewable_data.resize(exponent_size * n * num_outputs);
  auto out = exponents_viewable_data.data();
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    basdv::async_copy_to_device(
        basct::span<uint8_t>{out, n * exponent_size},
        basct::cspan<uint8_t>{exponents[output_index] + rng.a() * exponent_size, n * exponent_size},
        stream);
    out += n * exponent_size;
  }
  return exponents_viewable_data;
}
} // namespace sxt::mtxbk
