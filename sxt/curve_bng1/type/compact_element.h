#pragma once

#include "sxt/field25/type/element.h"
#include "sxt/field25/constant/one.h"

namespace sxt::cn1t {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
struct compact_element {
  f25t::element X;
  f25t::element Y;

  constexpr bool is_identity() const noexcept { return X[3] == static_cast<uint64_t>(-1); }

  static constexpr compact_element identity() noexcept {
    return {
        {0, 0, 0, static_cast<uint64_t>(-1)},
        f25cn::one_v,
    };
  }
};
} // namespace sxt::cn1t
