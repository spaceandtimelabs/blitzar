#pragma once

#include <memory>

#include "sxt/cbindings/base/curve_id.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor_base.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// multiexp_handle
//--------------------------------------------------------------------------------------------------
struct multiexp_handle {
  curve_id_t curve_id;
  std::unique_ptr<mtxpp2::partition_table_accessor_base> partition_table_accessor;
};
} // namespace sxt::cbnb

