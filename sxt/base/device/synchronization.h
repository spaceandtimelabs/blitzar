#pragma once

struct CUstream_st;

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// synchronize_stream
//--------------------------------------------------------------------------------------------------
void synchronize_stream(CUstream_st* stream) noexcept;
} // namespace sxt::basdv
