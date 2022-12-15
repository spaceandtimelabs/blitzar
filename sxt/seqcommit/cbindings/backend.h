#pragma once

#include "sxt/seqcommit/backend/pedersen_backend.h"
#include "sxt/seqcommit/cbindings/pedersen_capi.h"

namespace sxt::sqccb {

//--------------------------------------------------------------------------------------------------
// is_backend_initialized
//--------------------------------------------------------------------------------------------------
bool is_backend_initialized() noexcept;

//--------------------------------------------------------------------------------------------------
// get_backend
//--------------------------------------------------------------------------------------------------
sqcbck::pedersen_backend* get_backend() noexcept;

//--------------------------------------------------------------------------------------------------
// reset_backend_for_testing
//--------------------------------------------------------------------------------------------------
void reset_backend_for_testing() noexcept;

} // namespace sxt::sqccb
