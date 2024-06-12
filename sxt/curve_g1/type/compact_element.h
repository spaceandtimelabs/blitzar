#pragma once

#include "sxt/field12/constant/one.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/type/element.h"

namespace sxt::cg1t {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
struct compact_element {
  f12t::element X;
  f12t::element Y;

  constexpr bool is_identity() const noexcept {
    return X[5] == static_cast<uint64_t>(-1);
  }

  static constexpr compact_element identity() noexcept {
    return {
        {0, 0, 0, 0, 0, static_cast<uint64_t>(-1)},
        f12cn::one_v,
    };
  }
};
} // namespace sxt::cg1t
