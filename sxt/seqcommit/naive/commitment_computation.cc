#include "sxt/seqcommit/naive/commitment_computation.h"

#include <cassert>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/base/base_element.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/seqcommit/naive/reduce_exponent.h"

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// fill_data
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void fill_data(uint8_t a_i[32], const uint8_t *bytes_row_i_column_k, uint8_t size_row_data) noexcept {
  for (int j = 0; j < 32; ++j) {
    if (j < size_row_data) {
      // from (0) to (size_row_data - 1) we are populating a_i[j] with data values
      a_i[j] = bytes_row_i_column_k[j];
    } else {
      // padding zeros from (size_row_data) to (31)
      a_i[j] = 0;
    }
  }

  if (a_i[31] > 127) {
    reduce_exponent(a_i); // a_i = a_i % (2^252 + 27742317777372353535851937790883648493)
  }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept {

  assert(commitments.size() == value_sequences.size());

  for (size_t colum_k = 0; colum_k < commitments.size(); ++colum_k) {
    c21t::element_p3 p_k;
    const mtxb::exponent_sequence &column_k_data = value_sequences[colum_k];

    for (size_t row_i = 0; row_i < column_k_data.n; row_i++) {
      uint8_t a_i[32];
      c21t::element_p3 g_i;
      c21t::element_p3 h_i;
      uint8_t element_nbytes = column_k_data.element_nbytes;
      const uint8_t *bytes_row_i_column_k = column_k_data.data + row_i * element_nbytes;

      sqcb::compute_base_element(g_i, row_i);

      // fill a_i, inserting data values at the beginning and padding zeros at the end of a_i
      fill_data(a_i, bytes_row_i_column_k, element_nbytes);

      c21o::scalar_multiply(h_i, a_i, g_i); // h_i = a_i * g_i

      // aggregate all sum into p_k ==> p_k = p_k + a_i * g_i
      if (row_i == 0) {
        p_k = h_i; // initialize p_k
      } else {
        c21o::add(p_k, p_k, h_i);
      }
    }

    c21rs::to_bytes(commitments[colum_k].data(), p_k);
  }
}
}  // namespace sxt::sqcnv
