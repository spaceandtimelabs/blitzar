#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// generator_accessor 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
class generator_accessor {
 public:
   virtual ~generator_accessor() noexcept = default;

   virtual xena::future<> copy_precomputed_sums_to_device(basct::span<T> dest,
                                                          unsigned first) const noexcept = 0;
};
} // namespace sxt::mtxb
