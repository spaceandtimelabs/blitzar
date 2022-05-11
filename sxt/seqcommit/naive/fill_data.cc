#include "sxt/seqcommit/naive/fill_data.h"

#include "sxt/seqcommit/naive/reduce_exponent.h"

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// fill_data
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void fill_data(uint8_t a_i[32], const uint8_t *bytes_row_i_column_k, uint8_t size_row_data) noexcept {
    // #pragma unroll
    for (int j = 0; j < 32; ++j) {
        // from (0) to (size_row_data - 1) we are populating a_i[j] with data values
        // padding zeros from (size_row_data) to (31)
        a_i[j] = (j < size_row_data) ? bytes_row_i_column_k[j] : 0;
    }

    if (a_i[31] > 127) {
        reduce_exponent(a_i); // a_i = a_i % (2^252 + 27742317777372353535851937790883648493)
    }
}
}