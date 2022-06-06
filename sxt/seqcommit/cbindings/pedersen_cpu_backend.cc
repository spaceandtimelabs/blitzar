#include "sxt/seqcommit/cbindings/pedersen_cpu_backend.h"

#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/generator/cpu_generator.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

namespace sxt::sqccb {
//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void pedersen_cpu_backend::compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences,
    basct::span<sqcb::commitment> generators) noexcept {
    sqcnv::compute_commitments_cpu(commitments, value_sequences, generators);
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void pedersen_cpu_backend::get_generators(
    basct::span<sqcb::commitment> generators,
    uint64_t offset_generators) noexcept {
    sqcgn::cpu_get_generators(generators, offset_generators);
}

//--------------------------------------------------------------------------------------------------
// get_pedersen_cpu_backend
//--------------------------------------------------------------------------------------------------
pedersen_cpu_backend* get_pedersen_cpu_backend() {
    // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
    static pedersen_cpu_backend* backend = new pedersen_cpu_backend{};
    return backend;
}
}
