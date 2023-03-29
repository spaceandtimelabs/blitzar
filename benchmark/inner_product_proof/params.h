#pragma once

#include <memory>
#include <string>

#include "sxt/cbindings/backend/computational_backend.h"

namespace sxt::bncip {
//--------------------------------------------------------------------------------------------------
// params
//--------------------------------------------------------------------------------------------------
struct params {
  size_t n;
  size_t iterations;
  std::string backend_name;
  std::unique_ptr<cbnbck::computational_backend> backend;
};

//--------------------------------------------------------------------------------------------------
// read_params
//--------------------------------------------------------------------------------------------------
void read_params(params& params, int argc, char* argv[]) noexcept;
} // namespace sxt::bncip
