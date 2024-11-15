#include "sxt/proof/sumcheck/device_cache.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/memory/resource/device_resource.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// lookup
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
void device_cache::lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table,
                          basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept {
  auto& ptr = data_[basdv::get_device()];
  if (ptr == nullptr) {
    // TODO: copy data
  }
  product_table = ptr->product_table;
  product_terms = ptr->product_terms;
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
