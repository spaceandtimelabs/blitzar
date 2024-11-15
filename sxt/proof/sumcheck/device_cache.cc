#include "sxt/proof/sumcheck/device_cache.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/memory/resource/device_resource.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// make_device_copy 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
static std::unique_ptr<device_cache_data>
make_device_copy(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, basdv::stream& stream) noexcept {
  return nullptr;
}
#pragma clang diagnostic pop

//--------------------------------------------------------------------------------------------------
// lookup
//--------------------------------------------------------------------------------------------------
void device_cache::lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table,
                          basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept {
  auto& ptr = data_[basdv::get_device()];
  if (ptr == nullptr) {
    ptr = make_device_copy(product_table_, product_terms_, stream);
  }
  product_table = ptr->product_table;
  product_terms = ptr->product_terms;
}
} // namespace sxt::prfsk
