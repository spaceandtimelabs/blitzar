#include "sxt/seqcommit/cbindings/pedersen.h"

int sxt_init(const sxt_config* config) {
  (void)config;
  return 0;
}

int sxt_compute_pedersen_commitments(
    sxt_commitment* commitments, uint32_t num_sequences,
    const sxt_sequence_descriptor* descriptors) {
  (void)commitments;
  (void)num_sequences;
  (void)descriptors;
  return 0;
}
