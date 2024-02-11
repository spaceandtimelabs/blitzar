#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/device/stream.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// inclusive_prefix_count_buckets 
//--------------------------------------------------------------------------------------------------
void inclusive_prefix_count_buckets(basct::span<unsigned> counts, const basdv::stream& stream,
                                    basct::cspan<uint8_t> digits, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_outputs, unsigned n,
                                    unsigned num_tiles) noexcept {
  (void)counts;
  (void)stream;
  (void)digits;
  (void)element_num_bytes;
  (void)bit_width;
  (void)num_outputs;
  (void)n;
  (void)num_tiles;
}
} // namespace sxt::mtxbk
