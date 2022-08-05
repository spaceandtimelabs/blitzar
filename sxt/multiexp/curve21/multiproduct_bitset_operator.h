#pragma once

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// multiproduct_bitset_operator
//--------------------------------------------------------------------------------------------------
class multiproduct_bitset_operator {
  static constexpr uint64_t unset_marker_v = static_cast<uint64_t>(-1);

public:
  void mark_unset(c21t::element_p3& e) const noexcept { e.Z[4] = unset_marker_v; }

  bool is_set(const c21t::element_p3& e) const noexcept { return e.Z[4] != unset_marker_v; }

  void add(c21t::element_p3& res, const c21t::element_p3& lhs,
           const c21t::element_p3& rhs) const noexcept {
    c21o::add(res, lhs, rhs);
  }
};
} // namespace sxt::mtxc21
