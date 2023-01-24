#include "cbindings/pedersen.h"

#include <iostream>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// find_longest_sequence
//--------------------------------------------------------------------------------------------------
static uint64_t find_longest_sequence(uint64_t num_sequences,
                                      const sxt_sequence_descriptor* descriptors) {
  SXT_RELEASE_ASSERT(descriptors != nullptr);

  uint64_t longest_sequence = 0;
  for (uint32_t i = 0; i < num_sequences; ++i) {
    longest_sequence = std::max(longest_sequence, descriptors[i].n);
  }

  return longest_sequence;
}

//--------------------------------------------------------------------------------------------------
// populate_exponent_sequence
//--------------------------------------------------------------------------------------------------
static void populate_exponent_sequence(basct::span<sqcb::indexed_exponent_sequence> sequences,
                                       bool& has_sparse_sequence,
                                       const sxt_sequence_descriptor* descriptors) {
  SXT_RELEASE_ASSERT(descriptors != nullptr);

  has_sparse_sequence = false;

  for (uint32_t i = 0; i < sequences.size(); ++i) {
    auto& curr_descriptor = descriptors[i];

    SXT_RELEASE_ASSERT(curr_descriptor.n == 0 || curr_descriptor.data != nullptr);

    SXT_RELEASE_ASSERT(curr_descriptor.element_nbytes != 0 && curr_descriptor.element_nbytes <= 32);

    has_sparse_sequence |= (curr_descriptor.indices != nullptr);

    sequences[i] = {.exponent_sequence = {.element_nbytes = curr_descriptor.element_nbytes,
                                          .n = curr_descriptor.n,
                                          .data = curr_descriptor.data},
                    .indices = curr_descriptor.indices};
  }
}

//--------------------------------------------------------------------------------------------------
// process_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
static void process_compute_pedersen_commitments(struct sxt_compressed_ristretto* commitments,
                                                 uint32_t num_sequences,
                                                 const struct sxt_sequence_descriptor* descriptors,
                                                 basct::cspan<c21t::element_p3> generators) {
  SXT_RELEASE_ASSERT(commitments != nullptr);

  static_assert(sizeof(rstt::compressed_element) == sizeof(sxt_compressed_ristretto),
                "types must be ABI compatible");

  basct::span<rstt::compressed_element> commitments_result(
      reinterpret_cast<rstt::compressed_element*>(commitments), num_sequences);

  bool has_sparse_sequence;
  memmg::managed_array<sqcb::indexed_exponent_sequence> sequences(num_sequences);
  populate_exponent_sequence(sequences, has_sparse_sequence, descriptors);

  cbn::get_backend()->compute_commitments(commitments_result, sequences, generators,
                                          generators.size(), has_sparse_sequence);
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments_with_generators
//--------------------------------------------------------------------------------------------------
void sxt_compute_pedersen_commitments_with_generators(
    struct sxt_compressed_ristretto* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_ristretto* generators) {
  if (num_sequences == 0)
    return;

  SXT_RELEASE_ASSERT(generators != nullptr);
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized());

  uint64_t num_generators = find_longest_sequence(num_sequences, descriptors);
  basct::cspan<c21t::element_p3> generators_span(
      reinterpret_cast<const c21t::element_p3*>(generators), num_generators);

  process_compute_pedersen_commitments(commitments, num_sequences, descriptors, generators_span);
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
void sxt_compute_pedersen_commitments(sxt_compressed_ristretto* commitments, uint32_t num_sequences,
                                      const sxt_sequence_descriptor* descriptors,
                                      uint64_t offset_generators) {
  if (num_sequences == 0)
    return;

  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized());

  std::vector<c21t::element_p3> temp_generators;
  uint64_t num_generators = find_longest_sequence(num_sequences, descriptors);
  auto precomputed_generators = sxt::cbn::get_backend()->get_precomputed_generators(
      temp_generators, num_generators, offset_generators);

  process_compute_pedersen_commitments(commitments, num_sequences, descriptors,
                                       precomputed_generators);
}
