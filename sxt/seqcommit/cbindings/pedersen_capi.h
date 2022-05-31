#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SXT_DENSE_SEQUENCE_TYPE 1

#define SXT_BACKEND_CPU 1
#define SXT_BACKEND_GPU 2

/** config struct to hold the chosen backend **/
struct sxt_config {
  int backend;
};

/** describes a dense sequence of values **/
struct sxt_dense_sequence_descriptor {
  // the number of bytes used to represent an element in the sequence
  // element_nbytes must be a power of 2 and must satisfy
  //    1 <= element_nbytes <= 32
  uint8_t element_nbytes;

  // the number of elements in the sequence
  uint64_t n;

  // pointer to the data for the sequence of elements where there are n elements
  // in the sequence and each element enocodes a number of element_nbytes bytes
  // represented in the little endian format
  const uint8_t* data;
};

struct sxt_sequence_descriptor {
  // specifies the type of sequence (e.g. SXT_DENSE_SEQUENCE_TYPE, SXT_SPARSE_SEQUENCE_TYPE, etc).
  uint8_t sequence_type;

  union {
    struct sxt_dense_sequence_descriptor dense;

    // Note: we may also in the future want to support sparse sequences where
    // the majority of elements are zero and the nonzero elements in the
    // sequence are encoded with the pair (index, element)
  };
};

struct sxt_ristretto_element {
  // encodes an element of the ristretto255 group
  uint8_t ristretto_bytes[32];
};

/**
 * @brief Initialize the library. This should only be called once.
 * 
 * @param config [in] specifies which backend should be used in the computations (gpu or cpu)
 * @return int 
 */
int sxt_init(
  const struct sxt_config* config
);

/**
 * @brief Compute the pedersen commitments for sequences of values
 * 
 * Denote an element of a sequence by a_ij where i represents the sequence index
 * and j represents the element index. Let * represent the operator for the
 * ristretto255 group. Then res[i] encodes the ristretto255 group value
 * 
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * 
 * where n_i represents the number of elements in sequence i and g_j is a group
 * element determined by a prespecified function
 * 
 *     g: uint64_t -> ristretto255
 * 
 * @param commitments [out] an array of length num_sequences that provides the values of the
 *                         computed commitments for each sequence
 * 
 * @param num_sequences [in] specifies the number of sequences
 * @param descriptors   [in] an array of length num_sequences that specifies each sequence
 * 
 * @return 0 on success; otherwise a nonzero error code
 */
int sxt_compute_pedersen_commitments(
    struct sxt_ristretto_element* commitments,
    uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors
);

/**
 * @brief Gets the pre-specified random generated elements used for the Pedersen commitments
 * 
 * sxt_get_generators(generators, num_generators, offset_generators) → 
 *     generators[0] = generate_random_ristretto(0 + offset_generators)
 *     generators[1] = generate_random_ristretto(1 + offset_generators)
 *     generators[2] = generate_random_ristretto(2 + offset_generators)
 *       .
 *       .
 *       .
 *     generators[num_generators - 1] = generate_random_ristretto(num_generators - 1 + offset_generators)
 * 
 * @param generators [out] sxt_element_p3 pointer where the results must be written into
 * 
 * @param num_generators     [in] the total number of random generated elements to be computed
 * @param offset_generators  [in] the offset that shifts the first element computed from `0` to `offset_generators`
 * 
 * @return 0 on success; otherwise a nonzero error code
 */
int sxt_get_generators(
    struct sxt_ristretto_element* generators,
    uint64_t offset_generators,
    uint64_t num_generators
);

#ifdef __cplusplus
} // extern "C"
#endif

