#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T> class partition_table_accessor {
public:
  virtual ~partition_table_accessor() noexcept = default;

  virtual xena::future<> copy_precomputed_sums_to_device(basct::span<T> dest,
                                                         unsigned first) const noexcept = 0;
};
} // namespace sxt::mtxpp2
