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
#include "sxt/cbindings/backend/cpu_backend.h"

#include <cstring>
#include <vector>
#include <numeric>

#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/round_up.h"
#include "sxt/cbindings/base/curve_id_utility.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve_bng1/operation/add.h"
#include "sxt/curve_bng1/operation/double.h"
#include "sxt/curve_bng1/operation/neg.h"
#include "sxt/curve_bng1/type/conversion_utility.h"
#include "sxt/curve_bng1/type/element_affine.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/compression.h"
#include "sxt/curve_g1/operation/double.h"
#include "sxt/curve_g1/operation/neg.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/curve_gk/operation/add.h"
#include "sxt/curve_gk/operation/double.h"
#include "sxt/curve_gk/operation/neg.h"
#include "sxt/curve_gk/type/conversion_utility.h"
#include "sxt/curve_gk/type/element_affine.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve/multiexponentiation.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/proof_computation.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators) const noexcept {
  auto values = mtxcrv::compute_multiexponentiation<c21t::element_p3>(generators, value_sequences);
  rsto::batch_compress(commitments, values);
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<cg1t::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cg1t::element_p2> generators) const noexcept {
  auto values = mtxcrv::compute_multiexponentiation<cg1t::element_p2>(generators, value_sequences);
  cg1o::batch_compress(commitments, values);
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<cn1t::element_affine> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cn1t::element_p2> generators) const noexcept {
  auto values = mtxcrv::compute_multiexponentiation<cn1t::element_p2>(generators, value_sequences);
  cn1t::batch_to_element_affine(commitments, values);
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<cgkt::element_affine> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cgkt::element_p2> generators) const noexcept {
  auto values = mtxcrv::compute_multiexponentiation<cgkt::element_p2>(generators, value_sequences);
  cgkt::batch_to_element_affine(commitments, values);
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3>
cpu_backend::get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                                        uint64_t offset_generators) const noexcept {
  return sqcgn::get_precomputed_generators(temp_generators, n, offset_generators, false);
}

//--------------------------------------------------------------------------------------------------
// prove_inner_product
//--------------------------------------------------------------------------------------------------
void cpu_backend::prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                      basct::span<rstt::compressed_element> r_vector,
                                      s25t::element& ap_value, prft::transcript& transcript,
                                      const prfip::proof_descriptor& descriptor,
                                      basct::cspan<s25t::element> a_vector) const noexcept {
  prfip::cpu_driver drv;
  auto fut = prfip::prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor,
                                        a_vector);
  SXT_DEBUG_ASSERT(fut.ready());
}

//--------------------------------------------------------------------------------------------------
// verify_inner_product
//--------------------------------------------------------------------------------------------------
bool cpu_backend::verify_inner_product(prft::transcript& transcript,
                                       const prfip::proof_descriptor& descriptor,
                                       const s25t::element& product,
                                       const c21t::element_p3& a_commit,
                                       basct::cspan<rstt::compressed_element> l_vector,
                                       basct::cspan<rstt::compressed_element> r_vector,
                                       const s25t::element& ap_value) const noexcept {
  prfip::cpu_driver drv;
  return prfip::verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                     r_vector, ap_value)
      .value();
}

//--------------------------------------------------------------------------------------------------
// make_partition_table_accessor
//--------------------------------------------------------------------------------------------------
std::unique_ptr<mtxpp2::partition_table_accessor_base>
cpu_backend::make_partition_table_accessor(cbnb::curve_id_t curve_id, const void* generators,
                                           unsigned n) const noexcept {
  std::unique_ptr<mtxpp2::partition_table_accessor_base> res;
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        if constexpr (std::is_same_v<U, T>) {
          res = mtxpp2::make_in_memory_partition_table_accessor<T>(
              basct::cspan<T>{static_cast<const T*>(generators), n}, basm::alloc_t{});
        } else {
          res = mtxpp2::make_in_memory_partition_table_accessor<U, T>(
              basct::cspan<T>{static_cast<const T*>(generators), n}, basm::alloc_t{});
        }
      });
  return res;
}

//--------------------------------------------------------------------------------------------------
// fixed_multiexponentiation
//--------------------------------------------------------------------------------------------------
void cpu_backend::fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                            const mtxpp2::partition_table_accessor_base& accessor,
                                            unsigned element_num_bytes, unsigned num_outputs,
                                            unsigned n, const uint8_t* scalars) const noexcept {
  cbnb::switch_curve_type(curve_id, [&]<class U, class T>(std::type_identity<U>,
                                                          std::type_identity<T>) noexcept {
    basct::span<T> res_span{static_cast<T*>(res), num_outputs};
    basct::cspan<uint8_t> scalars_span{scalars, element_num_bytes * num_outputs * n};
    mtxpp2::multiexponentiate<T>(res_span,
                                 static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
                                 element_num_bytes, scalars_span);
  });
}

void cpu_backend::fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                            const mtxpp2::partition_table_accessor_base& accessor,
                                            const unsigned* output_bit_table, unsigned num_outputs,
                                            unsigned n, const uint8_t* scalars) const noexcept {
  cbnb::switch_curve_type(curve_id, [&]<class U, class T>(std::type_identity<U>,
                                                          std::type_identity<T>) noexcept {
    basct::span<T> res_span{static_cast<T*>(res), num_outputs};
    basct::cspan<unsigned> output_bit_table_span{output_bit_table, num_outputs};
    auto output_num_bytes =
        basn::round_up(std::accumulate(output_bit_table, output_bit_table + num_outputs, 0), 8);
    basct::cspan<uint8_t> scalars_span{scalars, output_num_bytes * n};
    mtxpp2::multiexponentiate<T>(res_span,
                                 static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
                                 output_bit_table_span, scalars_span);
  });
}

//--------------------------------------------------------------------------------------------------
// get_cpu_backend
//--------------------------------------------------------------------------------------------------
cpu_backend* get_cpu_backend() {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static cpu_backend* backend = new cpu_backend{};
  return backend;
}
} // namespace sxt::cbnbck
