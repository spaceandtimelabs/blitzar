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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SXT_CPU_BACKEND 1
#define SXT_GPU_BACKEND 2

#define SXT_CURVE_RISTRETTO255 0
#define SXT_CURVE_BLS_381 1
#define SXT_CURVE_BN_254 2

/** config struct to hold the chosen backend */
struct sxt_config {
  int backend;
  uint64_t num_precomputed_generators;
};

/** encodes an element of the `ristretto255` group */
struct sxt_ristretto255_compressed {
  uint8_t ristretto_bytes[32];
};

/** encodes an element of the `bls12-381` `G1` group in compressed form */
struct sxt_bls12_381_g1_compressed {
  uint8_t g1_bytes[48];
};

/** encodes an element of the finite field for `curve25519`
 *
 * modulo `(2^252 + 27742317777372353535851937790883648493)`
 */
struct sxt_curve25519_scalar {
  uint8_t bytes[32];
};

/** encodes a strobe-based transcript */
struct sxt_transcript {
  uint8_t bytes[203];
};

/** encodes an element of the `curve25519` group */
struct sxt_ristretto255 {
  uint64_t X[5];
  uint64_t Y[5];
  uint64_t Z[5];
  uint64_t T[5];
};

/** encodes an affine element of the `bls12-381` `G1` group */
struct sxt_bls12_381_g1 {
  uint64_t X[6];
  uint64_t Y[6];
};

/** encodes an affine element of the `bls12-381` `G1` group in projective form */
struct sxt_bls12_381_g1_p2 {
  uint64_t X[6];
  uint64_t Y[6];
  uint64_t Z[6];
};

/** encodes an affine element of the `bn254` `G1` group */
struct sxt_bn254_g1 {
  uint64_t X[4];
  uint64_t Y[4];
  uint8_t infinity;
};

/** encodes an affine element of the `bn254` `G1` group in projective form */
struct sxt_bn254_g1_p2 {
  uint64_t X[4];
  uint64_t Y[4];
  uint64_t Z[4];
};

/** describes a sequence of values */
struct sxt_sequence_descriptor {
  // The number of bytes used to represent an element in the sequence.
  // `element_nbytes` must be a power of `2` and must satisfy `1 <= element_nbytes <= 32`.
  uint8_t element_nbytes;

  // The number of elements in the sequence.
  uint64_t n;

  // Pointer to the data for the sequence of elements where there are `n` elements
  // in the sequence and each element encodes a number of `element_nbytes` bytes
  // represented in the little endian format.
  const uint8_t* data;

  // Whether the elements are signed.
  // Note: if signed, then `element_nbytes` must be `<= 16`.
  int is_signed;
};

/** resources for multiexponentiations with pre-specified generators */
struct sxt_multiexp_handle;

/**
 * Initializes the library.
 *
 * This should only be called once.
 *
 * # Arguments:
 *
 * - config (in): specifies which backend should be used in the computations. Those
 *   available are: `SXT_GPU_BACKEND`, and `SXT_CPU_BACKEND`.
 *
 * # Return:
 *
 * - `0` on success; otherwise a nonzero error code
 */
int sxt_init(const struct sxt_config* config);

/**
 * Compute the Pedersen commitments for sequences of values that internally generates `curve25519`
 * group elements.
 *
 * Denote an element of a sequence by `a_ij` where `i` represents the sequence index
 * and `j` represents the element index. Let `*` represent the operator for the
 * ristretto255 group. Then `res[i]` encodes the ristretto255 group value
 *
 * ```text
 *     Prod_{j=1 to n_i} g_{offset_generators + j} ^ a_ij
 * ```
 *
 * where `n_i` represents the number of elements in sequence `i` and `g_{offset_generators + j}`
 * is a group element determined by a prespecified function
 *
 * ```text
 *     g: uint64_t -> ristretto255
 * ```
 *
 * # Arguments:
 *
 * - `commitments` (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - `num_sequences` (in): specifies the number of sequences
 * - `descriptors` (in): an array of length `num_sequences` that specifies each sequence
 * - `offset_generators` (in): specifies the offset used to fetch the generators
 *
 * # Abnormal program termination in case of:
 *
 * - backend not initialized or incorrectly initialized
 * - `descriptors == nullptr`
 * - `commitments == nullptr`
 * - `descriptor[i].element_nbytes == 0`
 * - `descriptor[i].element_nbytes > 32`
 * - `descriptor[i].n > 0 && descriptor[i].data == nullptr`
 *
 * # Considerations:
 *
 * - `num_sequences == 0` will skip the computation
 */
void sxt_curve25519_compute_pedersen_commitments(struct sxt_ristretto255_compressed* commitments,
                                                 uint32_t num_sequences,
                                                 const struct sxt_sequence_descriptor* descriptors,
                                                 uint64_t offset_generators);

/**
 * Compute the Pedersen commitments for sequences of values using `curve25519` group elements.
 *
 * Denote an element of a sequence by `a_ij` where `i` represents the sequence index
 * and `j` represents the element index. Let `*` represent the operator for the
 * ristretto255 group. Then `res[i]` encodes the ristretto255 group value.
 *
 * ```text
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * ```
 *
 * where `n_i` represents the number of elements in sequence `i` and `g_j` is a group
 * element determined by the `generators[j]` user value given as input.
 *
 * # Arguments:
 *
 * - `commitments` (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - `num_sequences` (in): specifies the number of sequences
 * - `descriptors` (in): an array of length `num_sequences` that specifies each sequence
 * - `generators` (in): an array of length `max_num_rows` equals the maximum between all `n_i`
 *
 * # Abnormal program termination in case of:
 *
 * - backend not initialized or incorrectly initialized
 * - `descriptors == nullptr`
 * - `commitments == nullptr`
 * - `descriptor[i].element_nbytes == 0`
 * - `descriptor[i].element_nbytes > 32`
 * - `descriptor[i].n > 0 && descriptor[i].data == nullptr`
 *
 * # Considerations:
 *
 * - `num_sequences == 0` will skip the computation
 */
void sxt_curve25519_compute_pedersen_commitments_with_generators(
    struct sxt_ristretto255_compressed* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_ristretto255* generators);

/**
 * Compute the Pedersen commitments for sequences of values using `bls12-381` `G1` group elements.
 *
 * Denote an element of a sequence by `a_ij` where `i` represents the sequence index
 * and `j` represents the element index. Let `*` represent the operator for the
 * `bls12-381` `G1` group. Then `res[i]` encodes the `bls12-381` `G1` group value
 *
 * ```text
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * ```
 *
 * where `n_i` represents the number of elements in sequence `i` and `g_j` is a group
 * element determined by the `generators[j]` user value given as input
 *
 * # Arguments:
 *
 * - `commitments` (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - `num_sequences` (in): specifies the number of sequences
 * - `descriptors` (in): an array of length `num_sequences` that specifies each sequence
 * - `generators` (in): an array of length `max_num_rows` equals the maximum between all `n_i`
 *
 * # Abnormal program termination in case of:
 *
 * - backend not initialized or incorrectly initialized
 * - `descriptors == nullptr`
 * - `commitments == nullptr`
 * - `descriptor[i].element_nbytes == 0`
 * - `descriptor[i].element_nbytes > 32`
 * - `descriptor[i].n > 0 && descriptor[i].data == nullptr`
 *
 * # Considerations:
 *
 * - `num_sequences == 0` will skip the computation
 */
void sxt_bls12_381_g1_compute_pedersen_commitments_with_generators(
    struct sxt_bls12_381_g1_compressed* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_bls12_381_g1* generators);

/**
 * Compute the Pedersen commitments for sequences of values using `bn254` `G1` group elements.
 *
 * Denote an element of a sequence by `a_ij` where `i` represents the sequence index
 * and `j` represents the element index. Let `*` represent the operator for the
 * `bn254` `G1` group. Then `res[i]` encodes the `bn254` `G1` group value
 *
 * ```text
 *     Prod_{j=1 to n_i} g_j ^ a_ij
 * ```
 *
 * where `n_i` represents the number of elements in sequence `i` and `g_j` is a group
 * element determined by the `generators[j]` user value given as input
 *
 * # Arguments:
 *
 * - `commitments` (out): an array of length num_sequences where the computed commitments
 *                     of each sequence must be written into
 *
 * - `num_sequences` (in): specifies the number of sequences
 * - `descriptors` (in): an array of length `num_sequences` that specifies each sequence
 * - `generators` (in): an array of length `max_num_rows` equals the maximum between all `n_i`
 *
 * # Abnormal program termination in case of:
 *
 * - backend not initialized or incorrectly initialized
 * - `descriptors == nullptr`
 * - `commitments == nullptr`
 * - `descriptor[i].element_nbytes == 0`
 * - `descriptor[i].element_nbytes > 32`
 * - `descriptor[i].n > 0 && descriptor[i].data == nullptr`
 *
 * # Considerations:
 *
 * - `num_sequences == 0` will skip the computation
 */
void sxt_bn254_g1_uncompressed_compute_pedersen_commitments_with_generators(
    struct sxt_bn254_g1* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_bn254_g1* generators);

/**
 * Gets the pre-specified random generated elements used for the Pedersen commitments in the
 * `sxt_curve25519_compute_pedersen_commitments` function.
 *
 * ```text
 * sxt_ristretto255_get_generators(generators, num_generators, offset_generators) →
 *     generators[0] = generate_random_ristretto(0 + offset_generators)
 *     generators[1] = generate_random_ristretto(1 + offset_generators)
 *     generators[2] = generate_random_ristretto(2 + offset_generators)
 *       ...
 *     generators[num_generators - 1] = generate_random_ristretto(num_generators - 1 +
 * offset_generators)
 * ```
 * # Arguments:
 *
 * - `generators` (out): `sxt_ristretto255` pointer where the results must be written into
 * - `offset_generators` (in): the offset that shifts the first element computed from `0` to
 * `offset_generators`
 * - `num_generators` (in): the total number of random generated elements to be computed
 *
 * # Return:
 *
 * - `0` on success; otherwise a nonzero error code
 *
 * # Invalid input parameters, which generate error code:
 *
 * - `num_generators > 0 && generators == nullptr`
 *
 * # Considerations:
 *
 * - `num_generators == 0` will skip the computation
 */
int sxt_ristretto255_get_generators(struct sxt_ristretto255* generators, uint64_t offset_generators,
                                    uint64_t num_generators);

/**
 * Gets the `n`-th Ristretto point.
 *
 * The `n`-th Ristretto point is defined as:
 *
 * ```text
 * if n == 0:
 *    one_commit[0] = ristretto_identity;
 * else:
 *    one_commit[0] = g[0] + g[1] + ... + g[n - 1];
 * ```
 *
 * where
 *
 * ```text
 * struct sxt_ristretto255 ristretto_identity = {
 *    {0, 0, 0, 0, 0},
 *    {1, 0, 0, 0, 0},
 *    {1, 0, 0, 0, 0},
 *    {0, 0, 0, 0, 0},
 * };
 * ```
 *
 * and `g[i]` is the `i`-th generator provided by `sxt_ristretto255_get_generators` function at
 * offset `0`.
 *
 * # Return:
 *
 * - `0` on success; otherwise a nonzero error code
 *
 * # Invalid input parameters, which generate error code:
 *
 * - `one_commit == nullptr`
 */
int sxt_curve25519_get_one_commit(struct sxt_ristretto255* one_commit, uint64_t n);

/**
 * Creates an inner product proof.
 *
 * The proof is created with respect to the base `G`, provided by
 * `sxt_ristretto255_get_generators(G, generators_offset, 1ull << ceil(log2(n)))`.
 *
 * The `verifier` transcript is passed in as a parameter so that the
 * challenges depend on the *entire* transcript (including parent
 * protocols).
 *
 * Note that we don't have any restriction to the `n` value, other than
 * it has to be non-zero.
 *
 * # Algorithm description
 *
 * Initially, we compute `G` and `Q = G[np]`, where `np = 1ull << ceil(log2(n))`
 * and `G` is zero-indexed.
 *
 * The protocol consists of `k = ceil(lg_2(n))` rounds, indexed by `j = k - 1 , ... , 0`.
 *
 * In the `j`-th round, the prover computes:
 *
 * ```text
 * a_lo = {a[0], a[1], ..., a[n / 2 - 1]}
 * a_hi = {a[n/2], a[n/2 + 1], ..., a[n - 1]}
 * b_lo = {b[0], b[1], ..., b[n / 2 - 1]}
 * b_hi = {b[n/2], b[n/2 + 1], ..., b[n - 1]}
 * G_lo = {G[0], G[1], ..., G[n / 2 - 1]}
 * G_hi = {G[n/2], G[n/2 + 1], ..., G[n-1]}
 *
 * l_vector[j] = <a_lo, G_hi> + <a_lo, b_hi> * Q
 * r_vector[j] = <a_hi, G_lo> + <a_hi, b_lo> * Q
 * ```
 *
 * Note that if the `a` or `b` length is not a power of `2`,
 * then `a` or `b` is padded with zeros until it has a power of `2`.
 * `G` always has a power of `2` given how it is constructed.
 *
 * Then the prover sends `l_vector[j]` and `r_vector[j]` to the verifier,
 * and the verifier responds with a
 * challenge value `u[j] <- Z_p` (finite field of order p),
 * which is non-interactively simulated by
 * the input strobe-based transcript:
 *
 * ```text
 * transcript.append("L", l_vector[j]);
 * transcript.append("R", r_vector[j]);
 *
 * u[j] = transcript.challenge_value("x");
 * ```
 *
 * Then the prover uses `u[j]` to compute
 *
 * ```text
 * a = a_lo * u[j] + (u[j]^-1) * a_hi;
 * b = b_lo * (u[j]^-1) + u[j] * b_hi;
 * ```
 *
 * Then, the prover and verifier both compute
 *
 * ```text
 * G = G_lo * (u[j]^-1) + u[j] * G_hi
 *
 * n = n / 2;
 * ```
 *
 * and use these vectors (all of length `2^j`) for the next round.
 *
 * After the last `(j = 0)` round, the prover sends `ap_value = a[0]` to the verifier.
 *
 * # Arguments:
 *
 * - `l_vector` (out): transcript point array with length `ceil(log2(n))`
 * - `r_vector` (out): transcript point array with length `ceil(log2(n))`
 * - `ap_value` (out): a single scalar
 * - `transcript` (in/out): a single strobe-based transcript
 * - `n` (in): non-zero length for the input arrays
 * - `generators_offset` (in): offset used to fetch the bases
 * - `a_vector` (in): array with length `n`
 * - `b_vector` (in): array with length `n`
 *
 * # Abnormal program termination in case of:
 *
 * - `transcript`, `ap_value`, `b_vector`, or `a_vector` is `nullptr`
 * - `n` is zero
 * - `n` is non-zero, but `l_vector` or `r_vector` is `nullptr`
 */
void sxt_curve25519_prove_inner_product(struct sxt_ristretto255_compressed* l_vector,
                                        struct sxt_ristretto255_compressed* r_vector,
                                        struct sxt_curve25519_scalar* ap_value,
                                        struct sxt_transcript* transcript, uint64_t n,
                                        uint64_t generators_offset,
                                        const struct sxt_curve25519_scalar* a_vector,
                                        const struct sxt_curve25519_scalar* b_vector);

/**
 * Verifies an inner product proof.
 *
 * The proof is verified with respect to the base G, provided by
 * `sxt_ristretto255_get_generators(G, generators_offset, 1ull << ceil(log2(n)))`.
 *
 * Note that we don't have any restriction to the `n` value, other than
 * it has to be non-zero.
 *
 * # Arguments:
 *
 * - `transcript` (in/out): a single strobe-based transcript
 * - `n` (in): non-zero length for the input arrays
 * - `generators_offset` (in): offset used to fetch the bases
 * - `b_vector` (in): array with length `n`, the same one used by
 * `sxt_curve25519_prove_inner_product`
 * - `product` (in): a single scalar, represented by `<a, b>`,
 *                 the inner product of the two vectors `a` and `b` used by
 * `sxt_curve25519_prove_inner_product`
 * - `a_commit` (in): a single ristretto point, represented by `<a, G>` (the inner product of the
 * two vectors)
 * - `l_vector` (in): transcript point array with length `ceil(log2(n))`, generated by
 * `sxt_curve25519_prove_inner_product`
 * - `r_vector` (in): transcript point array with length `ceil(log2(n))`, generated by
 * `sxt_curve25519_prove_inner_product`
 * - `ap_value` (in): a single scalar, generated by `sxt_curve25519_prove_inner_product`
 *
 * # Return:
 *
 * - `1` in case the proof can be verified; otherwise, return `0`
 *
 * # Abnormal program termination in case of:
 *
 * - `transcript`, `ap_value`, `product`, `a_commit`, or `b_vector` is `nullptr`
 * - `n` is zero
 * - `n` is non-zero, but `l_vector` or `r_vector` is `nullptr`
 */
int sxt_curve25519_verify_inner_product(struct sxt_transcript* transcript, uint64_t n,
                                        uint64_t generators_offset,
                                        const struct sxt_curve25519_scalar* b_vector,
                                        const struct sxt_curve25519_scalar* product,
                                        const struct sxt_ristretto255* a_commit,
                                        const struct sxt_ristretto255_compressed* l_vector,
                                        const struct sxt_ristretto255_compressed* r_vector,
                                        const struct sxt_curve25519_scalar* ap_value);

/**
 * Create a handle for computing multiexponentiations using a fixed sequence of generators.
 *
 * Note: `generators` must match the type indicated by `curve_id`
 *
 * curve_id                        generators type
 * SXT_CURVE_RISTRETTO255          struct sxt_ristretto255*
 * SXT_CURVE_BLS_381               struct sxt_bls12_381_g1_p2*
 * SXT_CURVE_BN_254                struct sxt_bn254_g1_p2*
 */
struct sxt_multiexp_handle* sxt_multiexp_handle_new(unsigned curve_id, const void* generators,
                                                    unsigned n);

/**
 * Free resources for a multiexponentiation handle
 */
void sxt_multiexp_handle_free(struct sxt_multiexp_handle* handle);

/**
 * TODO(rnburn): fill me in
 */
void sxt_fixed_multiexponentiation(void* res, const struct sxt_multiexp_handle* handle,
                                   unsigned num_outputs, unsigned n, const void* scalars);

#ifdef __cplusplus
} // extern "C"
#endif
