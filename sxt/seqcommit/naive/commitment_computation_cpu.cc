#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

#include <cassert>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/operation/add.h"

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// compute_commitments_cpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_cpu(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences, basct::cspan<sqcb::commitment> generators) noexcept {

    assert(commitments.size() == value_sequences.size());

    for (size_t sequence_k = 0; sequence_k < commitments.size(); ++sequence_k) {
        c21t::element_p3 p_k = c21cn::zero_p3_v;
        const sqcb::indexed_exponent_sequence &column_k_data = value_sequences[sequence_k];

        for (size_t row_i = 0; row_i < column_k_data.exponent_sequence.n; row_i++) {
            c21t::element_p3 g_i;
            c21t::element_p3 h_i;
            uint8_t element_nbytes = column_k_data.exponent_sequence.element_nbytes;
            const uint8_t *bytes_row_i_column_k = column_k_data.exponent_sequence.data + row_i * element_nbytes;

            // verify if default generators should be used
            // otherwise, use the above dense representation
            if (generators.empty()) {
                size_t row_g_i = row_i;

                // verify if sparse representation should be used
                if (column_k_data.indices != nullptr) {
                    row_g_i = column_k_data.indices[row_i];
                }

                sqcgn::compute_base_element(g_i, row_g_i);
            } else { // otherwise, use the user given generators
                c21rs::from_bytes(g_i, generators[row_i].data());
            }

            c21o::scalar_multiply(
                h_i,
                basct::cspan<uint8_t>{bytes_row_i_column_k, element_nbytes},
                g_i);  // h_i = a_i * g_i

            // aggregate all sum into p_k ==> p_k = p_k + a_i * g_i
            c21o::add(p_k, p_k, h_i);
        }

        c21rs::to_bytes(commitments[sequence_k].data(), p_k);
    }
}

}  // namespace sxt::sqcnv
