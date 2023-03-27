#pragma once

#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/inner_product/workspace.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// gpu_workspace
//--------------------------------------------------------------------------------------------------
struct gpu_workspace final : public workspace {
  gpu_workspace() noexcept;

  size_t round_index;
  const proof_descriptor* descriptor;
  memmg::managed_array<s25t::element> a_vector;
  memmg::managed_array<s25t::element> b_vector;
  memmg::managed_array<c21t::element_p3> g_vector;

  // workspace
  xena::future<> ap_value(s25t::element& value) const noexcept override;
};
} // namespace sxt::prfip
