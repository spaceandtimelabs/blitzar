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
#include "sxt/cbindings/backend/gpu_backend.h"

#include <numeric>
#include <vector>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/system/directory_recorder.h"
#include "sxt/base/system/file_io.h"
#include "sxt/cbindings/backend/callback_sumcheck_transcript.h"
#include "sxt/cbindings/backend/computational_backend_utility.h"
#include "sxt/cbindings/base/curve_id_utility.h"
#include "sxt/cbindings/base/field_id_utility.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
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
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve/multiexponentiation.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/multiexp/pippenger2/multiexponentiation_serialization.h"
#include "sxt/multiexp/pippenger2/variable_length_multiexponentiation.h"
#include "sxt/proof/inner_product/gpu_driver.h"
#include "sxt/proof/inner_product/proof_computation.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/ristretto/type/literal.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"

using sxt::rstt::operator""_rs;

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// pre_initialize_gpu
//--------------------------------------------------------------------------------------------------
static void pre_initialize_gpu() noexcept {
  // A small dummy computation to avoid the future cost of JIT compiling PTX code
  memmg::managed_array<c21t::element_p3> generators = {
      0x123_rs,
  };
  memmg::managed_array<uint8_t> data = {1};
  memmg::managed_array<mtxb::exponent_sequence> value_sequences = {
      mtxb::exponent_sequence{
          .element_nbytes = 1,
          .n = 1,
          .data = data.data(),
      },
  };
  auto fut =
      mtxcrv::async_compute_multiexponentiation<c21t::element_p3>(generators, value_sequences);
  xens::get_scheduler().run();
}

//--------------------------------------------------------------------------------------------------
// gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend::gpu_backend() noexcept { pre_initialize_gpu(); }

//--------------------------------------------------------------------------------------------------
// prove_sumcheck
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
void gpu_backend::prove_sumcheck(void* polynomials, void* evaluation_point, unsigned field_id,
                                 const void* transcript_callback, void* transcript_context,
                                 const void* mles, const void* product_table,
                                 const unsigned* product_terms, unsigned num_outputs,
                                 unsigned n) noexcept {
  cbnb::switch_field_type(static_cast<cbnb::field_id_t>(field_id),
                          [&]<class T>(std::type_identity<T>) noexcept {
                            static_assert(std::same_as<T, s25t::element>,
                                          "only support curve-255 right now");
                            // fill me in
                            callback_sumcheck_transcript transcript{
                                reinterpret_cast<callback_sumcheck_transcript::callback_t>(
                                    const_cast<void*>(transcript_callback)),
                                transcript_context};
                          });
}
#pragma clang diagnostic pop

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators) const noexcept {
  auto fut =
      mtxcrv::async_compute_multiexponentiation<c21t::element_p3>(generators, value_sequences);
  xens::get_scheduler().run();
  rsto::batch_compress(commitments, fut.value());
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<cg1t::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cg1t::element_p2> generators) const noexcept {
  auto fut =
      mtxcrv::async_compute_multiexponentiation<cg1t::element_p2>(generators, value_sequences);
  xens::get_scheduler().run();
  cg1o::batch_compress(commitments, fut.value());
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<cn1t::element_affine> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cn1t::element_p2> generators) const noexcept {
  auto fut =
      mtxcrv::async_compute_multiexponentiation<cn1t::element_p2>(generators, value_sequences);
  xens::get_scheduler().run();
  cn1t::batch_to_element_affine(commitments, fut.value());
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<cgkt::element_affine> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<cgkt::element_p2> generators) const noexcept {
  auto fut =
      mtxcrv::async_compute_multiexponentiation<cgkt::element_p2>(generators, value_sequences);
  xens::get_scheduler().run();
  cgkt::batch_to_element_affine(commitments, fut.value());
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3>
gpu_backend::get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                                        uint64_t offset_generators) const noexcept {
  return sqcgn::get_precomputed_generators(temp_generators, n, offset_generators, true);
}

//--------------------------------------------------------------------------------------------------
// prove_inner_product
//--------------------------------------------------------------------------------------------------
void gpu_backend::prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                      basct::span<rstt::compressed_element> r_vector,
                                      s25t::element& ap_value, prft::transcript& transcript,
                                      const prfip::proof_descriptor& descriptor,
                                      basct::cspan<s25t::element> a_vector) const noexcept {
  prfip::gpu_driver drv;
  auto fut = prfip::prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor,
                                        a_vector);
  xens::get_scheduler().run();
  SXT_DEBUG_ASSERT(fut.ready());
}

//--------------------------------------------------------------------------------------------------
// verify_inner_product
//--------------------------------------------------------------------------------------------------
bool gpu_backend::verify_inner_product(prft::transcript& transcript,
                                       const prfip::proof_descriptor& descriptor,
                                       const s25t::element& product,
                                       const c21t::element_p3& a_commit,
                                       basct::cspan<rstt::compressed_element> l_vector,
                                       basct::cspan<rstt::compressed_element> r_vector,
                                       const s25t::element& ap_value) const noexcept {
  prfip::gpu_driver drv;
  auto fut = prfip::verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                         r_vector, ap_value);
  xens::get_scheduler().run();
  return fut.value();
}

//--------------------------------------------------------------------------------------------------
// make_partition_table_accessor
//--------------------------------------------------------------------------------------------------
std::unique_ptr<mtxpp2::partition_table_accessor_base>
gpu_backend::make_partition_table_accessor(cbnb::curve_id_t curve_id, const void* generators,
                                           unsigned n) const noexcept {
  std::unique_ptr<mtxpp2::partition_table_accessor_base> res;
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        if constexpr (std::is_same_v<U, T>) {
          res = mtxpp2::make_in_memory_partition_table_accessor<T>(
              basct::cspan<T>{static_cast<const T*>(generators), n});
        } else {
          res = mtxpp2::make_in_memory_partition_table_accessor<U, T>(
              basct::cspan<T>{static_cast<const T*>(generators), n});
        }
      });
  return res;
}

//--------------------------------------------------------------------------------------------------
// fixed_multiexponentiation
//--------------------------------------------------------------------------------------------------
void gpu_backend::fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                            const mtxpp2::partition_table_accessor_base& accessor,
                                            unsigned element_num_bytes, unsigned num_outputs,
                                            unsigned n, const uint8_t* scalars) const noexcept {
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        basct::span<T> res_span{static_cast<T*>(res), num_outputs};
        basct::cspan<uint8_t> scalars_span{scalars, element_num_bytes * num_outputs * n};
        auto fut = mtxpp2::async_multiexponentiate<T>(
            res_span, static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
            element_num_bytes, scalars_span);
        xens::get_scheduler().run();
      });
}

void gpu_backend::fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                            const mtxpp2::partition_table_accessor_base& accessor,
                                            const unsigned* output_bit_table, unsigned num_outputs,
                                            unsigned n, const uint8_t* scalars) const noexcept {
  cbnb::switch_curve_type(curve_id, [&]<class U, class T>(std::type_identity<U>,
                                                          std::type_identity<T>) noexcept {
    basct::span<T> res_span{static_cast<T*>(res), num_outputs};
    basct::cspan<unsigned> output_bit_table_span{output_bit_table, num_outputs};
    auto output_num_bytes =
        basn::divide_up(std::accumulate(output_bit_table, output_bit_table + num_outputs, 0), 8);
    basct::cspan<uint8_t> scalars_span{scalars, output_num_bytes * n};

    bassy::directory_recorder recorder{"packed-multiexponentiation"};
    if (recorder.recording()) {
      mtxpp2::write_multiexponentiation<T>(
          recorder.dir(), static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
          output_bit_table_span, scalars_span);
    }

    auto fut = mtxpp2::async_multiexponentiate<T>(
        res_span, static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
        output_bit_table_span, scalars_span);

    xens::get_scheduler().run();

    if (recorder.recording()) {
      bassy::write_file<T>(std::format("{}/result.bin", recorder.dir()), res_span);
    }
  });
}

void gpu_backend::fixed_multiexponentiation(void* res, cbnb::curve_id_t curve_id,
                                            const mtxpp2::partition_table_accessor_base& accessor,
                                            const unsigned* output_bit_table,
                                            const unsigned* output_lengths, unsigned num_outputs,
                                            const uint8_t* scalars) const noexcept {
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        basct::span<T> res_span{static_cast<T*>(res), num_outputs};
        basct::cspan<unsigned> output_bit_table_span{output_bit_table, num_outputs};
        basct::cspan<unsigned> output_lengths_span{output_lengths, num_outputs};
        auto scalars_span = make_scalars_span(scalars, output_bit_table_span, output_lengths_span);

        bassy::directory_recorder recorder{"vlen-multiexponentiation"};
        if (recorder.recording()) {
          mtxpp2::write_multiexponentiation<T>(
              recorder.dir(), static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
              output_bit_table_span, output_lengths_span, scalars_span);
        }

        auto fut = mtxpp2::async_multiexponentiate<T>(
            res_span, static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor),
            output_bit_table_span, output_lengths_span, scalars_span);

        xens::get_scheduler().run();

        if (recorder.recording()) {
          bassy::write_file<T>(std::format("{}/result.bin", recorder.dir()), res_span);
        }
      });
}

//--------------------------------------------------------------------------------------------------
// read_partition_table_accessor
//--------------------------------------------------------------------------------------------------
std::unique_ptr<mtxpp2::partition_table_accessor_base>
gpu_backend::read_partition_table_accessor(cbnb::curve_id_t curve_id,
                                           const char* filename) const noexcept {
  std::unique_ptr<mtxpp2::partition_table_accessor_base> res;
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        res = std::make_unique<mtxpp2::in_memory_partition_table_accessor<U>>(filename);
      });
  return res;
}

//--------------------------------------------------------------------------------------------------
// get_gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend* get_gpu_backend() noexcept {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static gpu_backend* backend = new gpu_backend{};
  return backend;
}
} // namespace sxt::cbnbck
