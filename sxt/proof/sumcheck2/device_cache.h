#pragma once

#include <utility>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/device/device_map.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/field/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// device_cache_data
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
struct device_cache_data {
  memmg::managed_array<std::pair<T, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
};

//--------------------------------------------------------------------------------------------------
// make_device_copy
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
std::unique_ptr<device_cache_data<T>>
make_device_copy(basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, basdv::stream& stream) noexcept {
  device_cache_data res{
      .product_table{product_table.size(), memr::get_device_resource()},
      .product_terms{product_terms.size(), memr::get_device_resource()},
  };
  basdv::async_copy_host_to_device(res.product_table, product_table, stream);
  basdv::async_copy_host_to_device(res.product_terms, product_terms, stream);
  return std::make_unique<device_cache_data>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// device_cache
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class device_cache {
public:
  device_cache(basct::cspan<std::pair<T, unsigned>> product_table,
               basct::cspan<unsigned> product_terms) noexcept;

  void lookup(basct::cspan<std::pair<T, unsigned>>& product_table,
              basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept;

  std::unique_ptr<device_cache_data<T>> clear() noexcept;

private:
  basct::cspan<std::pair<T, unsigned>> product_table_;
  basct::cspan<unsigned> product_terms_;
  basdv::device_map<std::unique_ptr<device_cache_data<T>>> data_;
};
} // namespace sxt::prfsk2
