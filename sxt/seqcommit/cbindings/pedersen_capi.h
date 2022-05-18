#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SXT_DENSE_SEQUENCE_TYPE 1

#define SXT_BACKEND_CPU 1
#define SXT_BACKEND_GPU 2

struct sxt_config {
  int backend;
};

// describes a dense sequence of values
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
    sxt_dense_sequence_descriptor dense;

    // Note: we may also in the future want to support sparse sequences where
    // the majority of elements are zero and the nonzero elements in the
    // sequence are encoded with the pair (index, element)
  };
};

struct sxt_commitment {
  // encodes an element of the ristretto255 group
  uint8_t ristretto_bytes[32];
};

// Initialize the exponentiation library. This can only be called once.
// Input parameters:
//    config
//      specifies which backend should be used in the computations (gpu or cpu)
// Return values:
//    0 on success; otherwise a nonzero error code
int sxt_init(const sxt_config* config);

// Compute the pedersen commitments for sequences of values
//
// Input parameters:
//  num_sequences
//      specifies the number of sequences
//  descriptors
//      an array of length num_sequences that specifies each sequence
// Output parameters:
//  commitments
//      an array of length num_sequences that provides the values of the
//      computed commitments for each sequence
// Return values:
//    0 on success; otherwise a nonzero error code
//
// Denote an element of a sequence by a_ij where i represents the sequence index
// and j represents the element index. Let * represent the operator for the
// ristretto255 group. Then res[i] encodes the ristretto255 group value
//
//    Prod_{j=1 to n_i} g_j ^ a_ij
//
//  where n_i represents the number of elements in sequence i and g_j is a group
//  element determined by a prespecified function
//
//     g: uint64_t -> ristretto255
int sxt_compute_pedersen_commitments(
    sxt_commitment* commitments, uint32_t num_sequences,
    const sxt_sequence_descriptor* descriptors);

#ifdef __cplusplus
} // extern "C"
#endif

