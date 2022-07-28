#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SXT_NAIVE_BACKEND_CPU 1
#define SXT_NAIVE_BACKEND_GPU 2
#define SXT_PIPPENGER_BACKEND_CPU 3

/** config struct to hold the chosen backend **/
struct sxt_config {
  int backend;
  uint64_t num_precomputed_generators;
};

struct sxt_compressed_ristretto {
  // encodes an element of the ristretto255 group
  uint8_t ristretto_bytes[32];
};

struct sxt_ristretto {
  // encodes an element of the curve255 group
  uint64_t X[5];
  uint64_t Y[5];
  uint64_t Z[5];
  uint64_t T[5];
};

/** describes a sequence of values **/
struct sxt_sequence_descriptor {
  // the number of bytes used to represent an element in the sequence
  // element_nbytes must be a power of 2 and must satisfy 1 <= element_nbytes <= 32
  uint8_t element_nbytes;

  // the number of elements in the sequence
  uint64_t n;

  // pointer to the data for the sequence of elements where there are n elements
  // in the sequence and each element enocodes a number of element_nbytes bytes
  // represented in the little endian format
  const uint8_t* data;

  // when `indices` is nullptr, then the sequence descriptor
  // represents a dense sequence. In this case, data[i] is
  // always tied with row i.
  // If `indices` is not nullptr, then the sequence represents
  // a sparse sequence such that `indices[i]` holds
  // the actual row_i in which data[i] is tied with. In case
  // indices is not nullptr, then `indices` must have
  // exactly `n` elements.
  const uint64_t* indices;
};

/**
 * Initialize the library. This should only be called once.
 *
 * Arguments:
 *
 * - config (in): specifies which backend should be used in the computations. Those
 *   available are: SXT_NAIVE_BACKEND_CPU, SXT_NAIVE_BACKEND_GPU, and SXT_PIPPENGER_BACKEND_CPU
 *
 * # Return:
 *
 * - 0 on success; otherwise a nonzero error code
 */
int sxt_init(const struct sxt_config* config);

/**
 * Compute the pedersen commitments for sequences of values
 *
 * Denote an element of a sequence by a_ij where i represents the sequence index
 * and j represents the element index. Let * represent the operator for the
 * ristretto255 group. Then res\[i] encodes the ristretto255 group value
 *
 * ```text
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * ```
 *
 * where n_i represents the number of elements in sequence i and g_j is a group
 * element determined by a prespecified function
 *
 * ```text
 *     g: uint64_t -> ristretto255
 * ```
 *
 * # Arguments:
 *
 * - commitments   (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - num_sequences (in): specifies the number of sequences
 * - descriptors   (in): an array of length num_sequences that specifies each sequence
 *
 * # Return:
 *
 * - 0 on success; otherwise a nonzero error code
 *
 * # Invalid input parameters, which generate error code:
 *
 * - backend not initialized or incorrectly initialized
 * - descriptors == nullptr
 * - commitments == nullptr
 * - descriptor\[i].element_nbytes == 0
 * - descriptor\[i].element_nbytes > 32
 * - descriptor\[i].n > 0 && descriptor\[i].data == nullptr
 *
 * # Considerations:
 *
 * - num_sequences equal to 0 will skip the computation
 */
int sxt_compute_pedersen_commitments(struct sxt_compressed_ristretto* commitments,
                                     uint32_t num_sequences,
                                     const struct sxt_sequence_descriptor* descriptors);

/**
 * Compute the pedersen commitments for sequences of values
 *
 * Denote an element of a sequence by a_ij where i represents the sequence index
 * and j represents the element index. Let * represent the operator for the
 * ristretto255 group. Then res\[i] encodes the ristretto255 group value
 *
 * ```text
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * ```
 *
 * where n_i represents the number of elements in sequence i and g_j is a group
 * element determined by the `generators\[i]` user value given as input
 *
 * # Arguments:
 *
 * - commitments   (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - num_sequences (in): specifies the number of sequences
 * - descriptors   (in): an array of length num_sequences that specifies each sequence
 * - generators    (in): an array of length `max_num_rows` = `the maximum between all n_i`
 *
 * # Return:
 *
 * - 0 on success; otherwise a nonzero error code
 *
 * # Invalid input parameters, which generate error code:
 *
 * - backend not initialized or incorrectly initialized
 * - descriptors == nullptr
 * - commitments == nullptr
 * - generators == nullptr
 * - descriptor\[i].element_nbytes == 0
 * - descriptor\[i].element_nbytes > 32
 * - descriptor\[i].n > 0 && descriptor\[i].data == nullptr
 *
 * # Considerations:
 *
 * - num_sequences equal to 0 will skip the computation
 */
int sxt_compute_pedersen_commitments_with_generators(
    struct sxt_compressed_ristretto* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, struct sxt_ristretto* generators);

/**
 * Gets the pre-specified random generated elements used for the Pedersen commitments in the
 * `sxt_compute_pedersen_commitments` function
 *
 * sxt_get_generators(generators, num_generators, offset_generators) →
 *     generators\[0] = generate_random_ristretto(0 + offset_generators)
 *     generators\[1] = generate_random_ristretto(1 + offset_generators)
 *     generators\[2] = generate_random_ristretto(2 + offset_generators)
 *       .
 *       .
 *       .
 *     generators\[num_generators - 1] = generate_random_ristretto(num_generators - 1 +
 * offset_generators)
 *
 * # Arguments:
 *
 * - generators         (out): sxt_element_p3 pointer where the results must be written into
 *
 * - num_generators     (in): the total number of random generated elements to be computed
 * - offset_generators  (in): the offset that shifts the first element computed from `0` to
 * `offset_generators`
 *
 * # Return:
 *
 * - 0 on success; otherwise a nonzero error code
 *
 * # Invalid input parameters, which generate error code:
 *
 * - num_generators > 0 && generators == nullptr
 *
 * # Considerations:
 *
 * - num_generators equal to 0 will skip the computation
 */
int sxt_get_generators(struct sxt_ristretto* generators, uint64_t offset_generators,
                       uint64_t num_generators);

#ifdef __cplusplus
} // extern "C"
#endif
