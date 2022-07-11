#include "sxt/seqcommit/backend/pippenger_cpu_backend.h"

#include <cassert>
#include <cstring>

#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/cpu_generator.h"
#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

namespace sxt::sqcbck {

//--------------------------------------------------------------------------------------------------
// populate_exponents_array
//--------------------------------------------------------------------------------------------------
// returns true if there is some sparse sequence in 
// the `value_sequences` span; otherwise returns false
static bool populate_exponents_array(
    uint64_t &longest_sequence_size,
    memmg::managed_array<mtxb::exponent_sequence> &exponents,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences) {
    
    longest_sequence_size = 0;

    for (size_t i = 0; i < value_sequences.size(); ++i) {
        if (value_sequences[i].indices != nullptr) {
            return true;
        }
        
        longest_sequence_size = std::max(longest_sequence_size,
                    value_sequences[i].exponent_sequence.n);

        exponents[i] = value_sequences[i].exponent_sequence;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void pippenger_cpu_backend::compute_commitments(
    basct::span<rstt::compressed_element> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
    basct::span<rstt::compressed_element> generators) noexcept {
    
    memmg::managed_array<mtxb::exponent_sequence>
                        exponents(value_sequences.size());

    uint64_t longest_sequence_size;
    
    bool is_sparse = populate_exponents_array( 
        longest_sequence_size, exponents, value_sequences);

    /////////////////////////////////////////////////////////
    // TODO
    /////////////////////////////////////////////////////////
    // for now, we use the naive cpu implementation
    // to process sparse sequences. But later, this should
    // be changed to use the pippenger implementation instead
    /////////////////////////////////////////////////////////
    if (is_sparse) {
        sqcnv::compute_commitments_cpu(commitments,
                        value_sequences, generators);
        return;
    }

    memmg::managed_array<rstt::compressed_element> inout(longest_sequence_size);

    if (generators.data() != nullptr) {
        // copy user provided generators to inout
        std::memcpy(inout.data(), generators.data(),
            longest_sequence_size * sizeof(rstt::compressed_element));
    } else {
        // fetch the default generators
        this->get_generators(inout, 0);
    }

    mtxrs::multiexponentiation_cpu_driver drv;

    mtxpi::compute_multiexponentiation(
        inout,
        drv,
        exponents
    );

    std::memcpy(commitments.data(), inout.data(),
            commitments.size() * sizeof(rstt::compressed_element));
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void pippenger_cpu_backend::get_generators(
    basct::span<rstt::compressed_element> generators,
    uint64_t offset_generators) noexcept {
    sqcgn::cpu_get_generators(generators, offset_generators);
}

//--------------------------------------------------------------------------------------------------
// get_pippenger_cpu_backend
//--------------------------------------------------------------------------------------------------
pippenger_cpu_backend* get_pippenger_cpu_backend() {
    // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
    static pippenger_cpu_backend* backend = new pippenger_cpu_backend{};
    return backend;
}
}
