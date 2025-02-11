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

#include "sxt/cbindings/backend/computational_backend.h"

#include "sxt/cbindings/base/curve_id_utility.h"
#include "sxt/cbindings/base/field_id_utility.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"
#include "sxt/proof/sumcheck/sumcheck_transcript.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// sumcheck_transcript
//--------------------------------------------------------------------------------------------------
namespace {
class sumcheck_transcript final : public prfsk::sumcheck_transcript {
public:
  using callback_t = void (*)(s25t::element* r, void* context, const s25t::element* polynomial,
                              unsigned round_degree);
  sumcheck_transcript(callback_t f, void* context) noexcept : f_{f}, context_{context} {}

  void init(size_t /*num_variables*/, size_t /*round_degree*/) noexcept override {}

  void round_challenge(s25t::element& r, basct::cspan<s25t::element> polynomial) noexcept override {
    f_(&r, context_, polynomial.data(), static_cast<unsigned>(polynomial.size()));
  }

private:
  callback_t f_;
  void* context_;
};
} // anonymous namespace

//--------------------------------------------------------------------------------------------------
// prove_sumcheck
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
void computational_backend::prove_sumcheck(void* polynomials, void* evaluation_point,
                                           unsigned field_id, const void* transcript_callback,
                                           void* transcript_context, const void* mles,
                                           const void* product_table, const unsigned* product_terms,
                                           unsigned num_outputs, unsigned n) noexcept {
  cbnb::switch_field_type(
      static_cast<cbnb::field_id_t>(field_id), [&]<class T>(std::type_identity<T>) noexcept {
        sumcheck_transcript transcript{reinterpret_cast<sumcheck_transcript::callback_t>(
                                           const_cast<void*>(transcript_callback)),
                                       transcript_context};
        (void)transcript;
      });
}
#pragma clang diagnostic pop

//--------------------------------------------------------------------------------------------------
// write_partition_table_accessor
//--------------------------------------------------------------------------------------------------
void computational_backend::write_partition_table_accessor(
    cbnb::curve_id_t curve_id, const mtxpp2::partition_table_accessor_base& accessor,
    const char* filename) const noexcept {
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor).write_to_file(filename);
      });
}
} // namespace sxt::cbnbck
