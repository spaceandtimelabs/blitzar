#pragma once

#include <cstddef>

#include "sxt/multiexp/pippenger/driver.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// test_driver
//--------------------------------------------------------------------------------------------------
class test_driver final : public driver {
public:
  // driver
  xena::future<memmg::managed_array<void>>
  compute_multiproduct(mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
                       const basct::blob_array& masks, size_t num_inputs) const noexcept override;

  xena::future<memmg::managed_array<void>>
  combine_multiproduct_outputs(xena::future<memmg::managed_array<void>>&& multiproduct,
                               basct::blob_array&& output_digit_or_all) const noexcept override;
};
} // namespace sxt::mtxpi
