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
#pragma once

#include <cinttypes>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/cbindings/backend/computational_backend.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// cpu_backend
//--------------------------------------------------------------------------------------------------
class cpu_backend final : public computational_backend {
public:
  void compute_commitments(basct::span<rstt::compressed_element> commitments,
                           basct::cspan<mtxb::exponent_sequence> value_sequences,
                           basct::cspan<c21t::element_p3> generators) const noexcept override;

  void compute_commitments(basct::span<cg1t::compressed_element> commitments,
                           basct::cspan<mtxb::exponent_sequence> value_sequences,
                           basct::cspan<cg1t::element_p2> generators) const noexcept override;

  void compute_commitments(basct::span<cn1t::element_affine> commitments,
                           basct::cspan<mtxb::exponent_sequence> value_sequences,
                           basct::cspan<cn1t::element_p2> generators) const noexcept override;

  void compute_commitments(basct::span<cgkt::element_affine> commitments,
                           basct::cspan<mtxb::exponent_sequence> value_sequences,
                           basct::cspan<cgkt::element_p2> generators) const noexcept override;

  basct::cspan<c21t::element_p3>
  get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                             uint64_t offset_generators) const noexcept override;

  void prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                           basct::span<rstt::compressed_element> r_vector, s25t::element& ap_value,
                           prft::transcript& transcript, const prfip::proof_descriptor& descriptor,
                           basct::cspan<s25t::element> a_vector) const noexcept override;

  bool verify_inner_product(prft::transcript& transcript, const prfip::proof_descriptor& descriptor,
                            const s25t::element& product, const c21t::element_p3& a_commit,
                            basct::cspan<rstt::compressed_element> l_vector,
                            basct::cspan<rstt::compressed_element> r_vector,
                            const s25t::element& ap_value) const noexcept override;

  std::unique_ptr<mtxpp2::partition_table_accessor_base>
  make_partition_table_accessor(cbnb::curve_id_t curve_id, const void* generators,
                                unsigned n) const noexcept override;

  void fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                 const mtxpp2::partition_table_accessor_base& accessor,
                                 unsigned element_num_bytes, unsigned num_outputs, unsigned n,
                                 const uint8_t* scalars) const noexcept override;

  void fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                 const mtxpp2::partition_table_accessor_base& accessor,
                                 const unsigned* output_bit_table, unsigned num_outputs, unsigned n,
                                 const uint8_t* scalars) const noexcept override;

  void fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                 const mtxpp2::partition_table_accessor_base& accessor,
                                 const unsigned* output_bit_table, const unsigned* output_lengths,
                                 unsigned num_outputs,
                                 const uint8_t* scalars) const noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_cpu_backend
//--------------------------------------------------------------------------------------------------
cpu_backend* get_cpu_backend();

} // namespace sxt::cbnbck
