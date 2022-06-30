#include "sxt/multiexp/test/compute_uint64_muladd.h"

#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_uint64_muladd
//--------------------------------------------------------------------------------------------------
void compute_uint64_muladd(
    basct::span<uint64_t> result,
    basct::span<uint64_t> generators,
    basct::span<mtxb::exponent_sequence> sequences) noexcept {
    
    // compute the expected result
    for (size_t seq = 0; seq < result.size(); ++seq) {
      result[seq] = 0;
      
      uint8_t element_nbytes = sequences[seq].element_nbytes;

      // sum all the elements in the current sequence together
      for (size_t gen_i = 0; gen_i < sequences[seq].n; ++gen_i) {
        uint64_t pow256 = 1;
        uint64_t curr_exponent = 0;
        
        // reconstructs the gen_i-th data element out of its element_nbytes values
        for (size_t j = 0; j < element_nbytes; ++j) {
          curr_exponent += sequences[seq].data[gen_i * element_nbytes + j] * pow256;
          pow256 *= 256;
        }
        
        result[seq] += generators[gen_i] * curr_exponent;
      }
    }
}
}  // namespace sxt::mtxtst
