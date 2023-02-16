#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basct {
class blob_array;
class span_cvoid;
} // namespace sxt::basct

namespace sxt::mtxb {
class exponent;
}
namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
public:
  virtual ~driver() noexcept = default;

  virtual xena::future<memmg::managed_array<void>>
  compute_multiproduct(mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
                       const basct::blob_array& masks, size_t num_inputs) const noexcept = 0;

  virtual xena::future<memmg::managed_array<void>>
  combine_multiproduct_outputs(xena::future<memmg::managed_array<void>>&& multiproduct,
                               basct::blob_array&& output_digit_or_all) const noexcept = 0;
};
} // namespace sxt::mtxpi
