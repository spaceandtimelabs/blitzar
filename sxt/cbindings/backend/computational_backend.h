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
#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/cbindings/base/curve_id.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor_base.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::rstt {
class compressed_element;
}

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::cn1t {
struct element_affine;
struct element_p2;
} // namespace sxt::cn1t

namespace sxt::cg1t {
class compressed_element;
struct element_p2;
} // namespace sxt::cg1t

namespace sxt::s25t {
class element;
}

namespace sxt::prft {
class transcript;
}

namespace sxt::prfip {
struct proof_descriptor;
}

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// computational_backend
//--------------------------------------------------------------------------------------------------
class computational_backend {
public:
  virtual ~computational_backend() noexcept = default;

  virtual void compute_commitments(basct::span<rstt::compressed_element> commitments,
                                   basct::cspan<mtxb::exponent_sequence> value_sequences,
                                   basct::cspan<c21t::element_p3> generators) const noexcept = 0;

  virtual void compute_commitments(basct::span<cg1t::compressed_element> commitments,
                                   basct::cspan<mtxb::exponent_sequence> value_sequences,
                                   basct::cspan<cg1t::element_p2> generators) const noexcept = 0;

  virtual void compute_commitments(basct::span<cn1t::element_affine> commitments,
                                   basct::cspan<mtxb::exponent_sequence> value_sequences,
                                   basct::cspan<cn1t::element_p2> generators) const noexcept = 0;

  virtual basct::cspan<c21t::element_p3>
  get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                             uint64_t offset_generators) const noexcept = 0;

  virtual void prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                   basct::span<rstt::compressed_element> r_vector,
                                   s25t::element& ap_value, prft::transcript& transcript,
                                   const prfip::proof_descriptor& descriptor,
                                   basct::cspan<s25t::element> a_vector) const noexcept = 0;

  virtual bool verify_inner_product(prft::transcript& transcript,
                                    const prfip::proof_descriptor& descriptor,
                                    const s25t::element& product, const c21t::element_p3& a_commit,
                                    basct::cspan<rstt::compressed_element> l_vector,
                                    basct::cspan<rstt::compressed_element> r_vector,
                                    const s25t::element& ap_value) const noexcept = 0;

  virtual std::unique_ptr<mtxpp2::partition_table_accessor_base>
  make_partition_table_accessor(cbnb::curve_id_t curve_id, const void* generators,
                                unsigned n) const noexcept = 0;

  virtual void fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                         const mtxpp2::partition_table_accessor_base& accessor,
                                         unsigned element_num_bytes, unsigned num_outputs,
                                         unsigned n, const uint8_t* scalars) const noexcept = 0;

  virtual void fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                         const mtxpp2::partition_table_accessor_base& accessor,
                                         basct::cspan<unsigned> output_bit_table, unsigned n,
                                         const uint8_t* scalars) const noexcept = 0;
};
} // namespace sxt::cbnbck
