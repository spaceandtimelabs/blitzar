#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_exponents_viewable
//--------------------------------------------------------------------------------------------------
basct::cspan<const uint8_t*>
make_exponents_viewable(memmg::managed_array<uint8_t>& exponents_viewable_data,
                        basct::cspan<const uint8_t*> exponents,
                        const basit::index_range& rng) noexcept {
  (void)exponents_viewable_data;
  (void)exponents;
  (void)rng;
  return {};
}
} // namespace sxt::mtxbk
