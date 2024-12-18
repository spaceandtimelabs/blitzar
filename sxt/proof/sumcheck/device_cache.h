#pragma once

#include <utility>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/device/device_map.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::basdv { class stream; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// device_cache_data
//--------------------------------------------------------------------------------------------------
struct device_cache_data {
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
};

//--------------------------------------------------------------------------------------------------
// device_cache
//--------------------------------------------------------------------------------------------------
class device_cache {
  public:
    device_cache(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms) noexcept;

    void lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table,
                basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept;

    std::unique_ptr<device_cache_data> clear() noexcept;

  private:
    basct::cspan<std::pair<s25t::element, unsigned>> product_table_;
    basct::cspan<unsigned> product_terms_;
    basdv::device_map<std::unique_ptr<device_cache_data>> data_;
};
} // namespace sxt::prfsk
