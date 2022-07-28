#include "sxt/seqcommit/cbindings/pedersen.h"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/backend/naive_cpu_backend.h"
#include "sxt/seqcommit/backend/naive_gpu_backend.h"
#include "sxt/seqcommit/backend/pedersen_backend.h"
#include "sxt/seqcommit/backend/pippenger_cpu_backend.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// backend
//--------------------------------------------------------------------------------------------------
static sqcbck::pedersen_backend* backend = nullptr;

//--------------------------------------------------------------------------------------------------
// get_num_devices
//--------------------------------------------------------------------------------------------------
static int get_num_devices() {
  int num_devices;

  auto rcode = cudaGetDeviceCount(&num_devices);

  if (rcode != cudaSuccess) {
    num_devices = 0;

    std::cout << "cudaGetDeviceCount failed: " << cudaGetErrorString(rcode) << "\n";
  }

  return num_devices;
}

//--------------------------------------------------------------------------------------------------
// sxt_init
//--------------------------------------------------------------------------------------------------
int sxt_init(const sxt_config* config) {
  if (config == nullptr)
    exit(1);
  if (backend != nullptr)
    exit(1);

  if (config->backend == SXT_NAIVE_BACKEND_CPU) {
    backend = sqcbck::get_naive_cpu_backend();

    sqcgn::init_precomputed_generators(config->num_precomputed_generators, false);

    return 0;
  } else if (config->backend == SXT_NAIVE_BACKEND_GPU) {
    int num_devices = get_num_devices();

    if (num_devices > 0) {
      backend = sqcbck::get_naive_gpu_backend();

      sqcgn::init_precomputed_generators(config->num_precomputed_generators, true);
    } else {
      backend = sqcbck::get_naive_cpu_backend();

      sqcgn::init_precomputed_generators(config->num_precomputed_generators, false);

      std::cout << "WARN: Using 'compute_commitments_cpu'. " << std::endl;
    }

    return 0;
  } else if (config->backend == SXT_PIPPENGER_BACKEND_CPU) {
    backend = sqcbck::get_pippenger_cpu_backend();

    sqcgn::init_precomputed_generators(config->num_precomputed_generators, false);

    return 0;
  }

  return 1;
}

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
                                                struct sxt_ristretto* generators) {

  // backend not initialized (sxt_init not called correctly)
  if (backend == nullptr)
    return 1;

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

  basct::cspan<c21t::element_p3> generators_span(reinterpret_cast<c21t::element_p3*>(generators),
                                                 generators_length);

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
    const struct sxt_sequence_descriptor* descriptors, struct sxt_ristretto* generators) {

  // we only verify if generators is null, but we
  // expect it to have size equal to the longest
  // row in sequence
  if (generators == nullptr)
    return 1;

  int ret =
      process_compute_pedersen_commitments(commitments, num_sequences, descriptors, generators);

  return ret;
}

//--------------------------------------------------------------------------------------------------
// sxt_get_generators
//--------------------------------------------------------------------------------------------------
int sxt_get_generators(struct sxt_ristretto* generators, uint64_t num_generators,
                       uint64_t offset_generators) {

  // if no generator specified, then ignore the function call
  if (num_generators == 0)
    return 0;

  // at least one generator to be computed, but null array pointer
  if (num_generators > 0 && generators == nullptr)
    return 1;

  // backend not initialized (sxt_init not called correctly)
  if (backend == nullptr)
    return 1;

  basct::span<c21t::element_p3> generators_result(reinterpret_cast<c21t::element_p3*>(generators),
                                                  num_generators);

  backend->get_generators(generators_result, offset_generators);

  return 0;
}

namespace sxt::sqccb {
//--------------------------------------------------------------------------------------------------
// reset_backend_for_testing
//--------------------------------------------------------------------------------------------------
int reset_backend_for_testing() {
  if (backend == nullptr)
    exit(1);

  backend = nullptr;

  return 0;
}
} // namespace sxt::sqccb
