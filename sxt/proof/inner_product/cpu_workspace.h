#pragma once

#include <cstddef>
#include <memory_resource>

#include "sxt/base/container/span.h"
#include "sxt/proof/inner_product/workspace.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
struct element;
}

namespace sxt::prfip {
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// cpu_workspace
//--------------------------------------------------------------------------------------------------
struct cpu_workspace final : public workspace {
  std::pmr::monotonic_buffer_resource alloc;
  size_t round_index;
  const proof_descriptor* descriptor;
  basct::cspan<s25t::element> a_vector0;
  basct::span<c21t::element_p3> g_vector;
  basct::span<s25t::element> a_vector;
  basct::span<s25t::element> b_vector;

  // workspace
  void ap_value(s25t::element& value) const noexcept override;
};
} // namespace sxt::prfip
