#include "benchmark/multi_commitment/multi_commitment_cpu.h"

#include <cstdio>

#include "sxt/base/container/span.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/seqcommit/naive/commitment_computation.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multi_commitment_cpu
//--------------------------------------------------------------------------------------------------
void multi_commitment_cpu(
    memmg::managed_array<sqcb::commitment> &commitments_per_col,
    uint64_t rows, uint64_t cols, uint64_t element_nbytes,
    const memmg::managed_array<uint8_t> &data_table) noexcept {

    memmg::managed_array<mtxb::exponent_sequence> data_cols(cols);

    for (size_t c = 0; c < cols; ++c) {
        auto &data_col = data_cols[c];

        data_col.n = rows;
        data_col.element_nbytes = element_nbytes;
        data_col.data = (data_table.data() + c * rows * element_nbytes);
    }

    basct::span<sqcb::commitment> commitments(commitments_per_col.data(), cols);
    basct::cspan<mtxb::exponent_sequence> value_sequences(data_cols.data(), cols);

    sqcnv::compute_commitments(commitments, value_sequences);
}
} // namespace sxt
