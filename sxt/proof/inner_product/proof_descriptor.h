#pragma once

#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// proof_descriptor
//--------------------------------------------------------------------------------------------------
/**
 * Description of inputs used for an inner product proof.
 *
 * In our version of an inner product proof, the prover is establishing that
 *    <a_vector, b_vector> = product
 * where the commitment
 *      commit_a = sum_i a[i] * g[i]
 * and b_vector are known to the verifier.
 */
struct proof_descriptor {
  basct::cspan<s25t::element> b_vector;
  basct::cspan<c21t::element_p3> g_vector;
  const c21t::element_p3* q_value = nullptr;
};
} // namespace sxt::prfip
