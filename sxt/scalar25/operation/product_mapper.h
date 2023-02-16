#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// product_mapper
//--------------------------------------------------------------------------------------------------
class product_mapper {
public:
  using value_type = s25t::element;

  CUDA_CALLABLE product_mapper(const s25t::element* lhs_data,
                               const s25t::element* rhs_data) noexcept
      : lhs_data_{lhs_data}, rhs_data_{rhs_data} {}

  CUDA_CALLABLE void map_index(s25t::element& res, unsigned int index) const noexcept {
    s25o::mul(res, lhs_data_[index], rhs_data_[index]);
  }

  CUDA_CALLABLE s25t::element map_index(unsigned int index) const noexcept {
    s25t::element res;
    s25o::mul(res, lhs_data_[index], rhs_data_[index]);
    return res;
  }

  CUDA_CALLABLE const s25t::element* lhs_data() const noexcept { return lhs_data_; }

  CUDA_CALLABLE const s25t::element* rhs_data() const noexcept { return rhs_data_; }

private:
  const s25t::element* lhs_data_;
  const s25t::element* rhs_data_;
};
} // namespace sxt::s25o
