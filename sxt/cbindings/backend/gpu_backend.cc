#include "sxt/cbindings/backend/gpu_backend.h"

#include <vector>

#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve21/multiexponentiation.h"
#include "sxt/proof/inner_product/cpu_driver.h"
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
  auto fut = mtxc21::async_compute_multiexponentiation(generators, value_sequences);
  xens::get_scheduler().run();
}

//--------------------------------------------------------------------------------------------------
// gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend::gpu_backend() noexcept { pre_initialize_gpu(); }

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators) const noexcept {
  auto fut = mtxc21::async_compute_multiexponentiation(generators, value_sequences);
  xens::get_scheduler().run();
  rsto::batch_compress(commitments, fut.value());
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
  // TODO: update this to use gpu_driver when available
  prfip::cpu_driver drv;
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
  // TODO: update this to use gpu_driver when available
  prfip::cpu_driver drv;
  auto fut = prfip::verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                         r_vector, ap_value);
  xens::get_scheduler().run();
  return fut.value();
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
