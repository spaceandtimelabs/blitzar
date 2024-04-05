#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T> class partition_table_accessor {
public:
  virtual ~partition_table_accessor() noexcept = default;

  virtual void async_copy_precomputed_sums_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                                                     unsigned first) const noexcept = 0;
};
} // namespace sxt::mtxpp2
