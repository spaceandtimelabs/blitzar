#include "sxt/seqcommit/cbindings/pedersen.h"

#include <iostream>

#include "sxt/memory/management/managed_array.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/cbindings/backend.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// validate_descriptor
//--------------------------------------------------------------------------------------------------
static int validate_descriptor(uint64_t& longest_sequence,
                               const sxt_sequence_descriptor* descriptor) {
  // verify if data pointers are validate
  if (descriptor->n > 0 && descriptor->data == nullptr)
    return 1;

  // verify if word size is inside the correct range (1 to 32)
  if (descriptor->element_nbytes == 0 || descriptor->element_nbytes > 32)
    return 1;

  longest_sequence = std::max(longest_sequence, descriptor->n);

  return 0;
}

//--------------------------------------------------------------------------------------------------
// validate_sequence_descriptor
//--------------------------------------------------------------------------------------------------
static int validate_sequence_descriptor(uint64_t& longest_sequence, uint32_t num_sequences,
                                        const sxt_sequence_descriptor* descriptors) {

  longest_sequence = 0;

  // invalid pointers
  if (descriptors == nullptr)
    return 1;

  // verify if each descriptor is inside the ranges
  for (uint32_t commit_index = 0; commit_index < num_sequences; ++commit_index) {
    if (validate_descriptor(longest_sequence, descriptors + commit_index))
      return 1;
  }

  return 0;
}

//--------------------------------------------------------------------------------------------------
// process_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
static int process_compute_pedersen_commitments(struct sxt_compressed_ristretto* commitments,
                                                uint32_t num_sequences,
                                                const struct sxt_sequence_descriptor* descriptors,
                                                const struct sxt_ristretto* generators) {

  // backend not initialized (sxt_init not called correctly)
  if (!sxt::sqccb::is_backend_initialized()) {
    std::cerr
        << "ABORT: backend uninitialized in the `process_compute_pedersen_commitments` function"
        << std::endl;
    std::abort();
  }

  if (num_sequences == 0)
    return 0;

  uint64_t length_longest_sequence = 0;

  // verify if input is invalid
  if (commitments == nullptr ||
      validate_sequence_descriptor(length_longest_sequence, num_sequences, descriptors))
    return 1;

  static_assert(sizeof(rstt::compressed_element) == sizeof(sxt_compressed_ristretto),
                "types must be ABI compatible");

  basct::span<rstt::compressed_element> commitments_result(
      reinterpret_cast<rstt::compressed_element*>(commitments), num_sequences);

  memmg::managed_array<sqcb::indexed_exponent_sequence> sequences(num_sequences);

  static_assert(sizeof(sqcb::indexed_exponent_sequence) == sizeof(sxt_sequence_descriptor),
                "types must be ABI compatible");

  bool has_sparse_sequence = false;

  // populating sequence object
  for (uint64_t i = 0; i < num_sequences; ++i) {
    has_sparse_sequence |= (descriptors[i].indices != nullptr);

    sequences[i] = *(reinterpret_cast<const sqcb::indexed_exponent_sequence*>(&descriptors[i]));
  }

  uint64_t generators_length = length_longest_sequence;
  basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences.data(), num_sequences);

  if (generators == nullptr) {
    generators_length = 0;
  }

  // we assume that the `compute_commitments` does not change any content in the generators span
  basct::cspan<c21t::element_p3> generators_span(
      reinterpret_cast<const c21t::element_p3*>(generators), generators_length);

  auto backend = sqccb::get_backend();
  backend->compute_commitments(commitments_result, value_sequences, generators_span,
                               length_longest_sequence, has_sparse_sequence);

  return 0;
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
int sxt_compute_pedersen_commitments(sxt_compressed_ristretto* commitments, uint32_t num_sequences,
                                     const sxt_sequence_descriptor* descriptors) {

  int ret = process_compute_pedersen_commitments(commitments, num_sequences, descriptors, nullptr);

  return ret;
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
int sxt_compute_pedersen_commitments_with_generators(
    struct sxt_compressed_ristretto* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_ristretto* generators) {

  // we only verify if generators is null, but we
  // expect it to have size equal to the longest
  // row in sequence
  if (generators == nullptr)
    return 1;

  int ret =
      process_compute_pedersen_commitments(commitments, num_sequences, descriptors, generators);

  return ret;
}
