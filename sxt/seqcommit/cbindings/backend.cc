#include "sxt/seqcommit/cbindings/backend.h"

#include <cassert>
#include <iostream>

#include "sxt/base/device/property.h"
#include "sxt/seqcommit/backend/naive_cpu_backend.h"
#include "sxt/seqcommit/backend/naive_gpu_backend.h"
#include "sxt/seqcommit/backend/pedersen_backend.h"
#include "sxt/seqcommit/backend/pippenger_cpu_backend.h"
#include "sxt/seqcommit/generator/precomputed_initializer.h"

using namespace sxt;

namespace sxt::sqccb {
//--------------------------------------------------------------------------------------------------
// backend
//--------------------------------------------------------------------------------------------------
static sqcbck::pedersen_backend* backend = nullptr;

//--------------------------------------------------------------------------------------------------
// initialize_naive_cpu_backend
//--------------------------------------------------------------------------------------------------
static void initialize_naive_cpu_backend(const sxt_config* config) noexcept {
  backend = sqcbck::get_naive_cpu_backend();
  sqcgn::init_precomputed_components(config->num_precomputed_generators, false);
}

//--------------------------------------------------------------------------------------------------
// initialize_pippenger_cpu_backend
//--------------------------------------------------------------------------------------------------
static void initialize_pippenger_cpu_backend(const sxt_config* config) noexcept {
  backend = sqcbck::get_pippenger_cpu_backend();
  sqcgn::init_precomputed_components(config->num_precomputed_generators, false);
}

//--------------------------------------------------------------------------------------------------
// initialize_naive_gpu_backend
//--------------------------------------------------------------------------------------------------
static void initialize_naive_gpu_backend(const sxt_config* config) noexcept {
  int num_devices = basdv::get_num_devices();

  if (num_devices == 0) {
    initialize_pippenger_cpu_backend(config);

    // this message is used only to warn the user that gpu will not be used
    std::cout << "WARN: Using pippenger cpu instead of naive gpu backend." << std::endl;

    return;
  }

  backend = sqcbck::get_naive_gpu_backend();
  sqcgn::init_precomputed_components(config->num_precomputed_generators, true);
}

//--------------------------------------------------------------------------------------------------
// is_backend_initialized
//--------------------------------------------------------------------------------------------------
bool is_backend_initialized() noexcept { return backend != nullptr; }

//--------------------------------------------------------------------------------------------------
// get_backend
//--------------------------------------------------------------------------------------------------
sqcbck::pedersen_backend* get_backend() noexcept {
  assert(backend != nullptr);

  return backend;
}

//--------------------------------------------------------------------------------------------------
// reset_backend_for_testing
//--------------------------------------------------------------------------------------------------
void reset_backend_for_testing() noexcept {
  assert(backend != nullptr);

  backend = nullptr;
}
} // namespace sxt::sqccb

//--------------------------------------------------------------------------------------------------
// sxt_init
//--------------------------------------------------------------------------------------------------
int sxt_init(const sxt_config* config) {
  if (config == nullptr) {
    std::cerr << "ABORT: config input to `sxt_init` c binding function is null" << std::endl;
    std::abort();
  }

  if (sqccb::backend != nullptr) {
    std::cerr << "ABORT: trying to reinitialize the backend in the `sxt_init` c binding function"
              << std::endl;
    std::abort();
  }

  if (config->backend == SXT_NAIVE_BACKEND_CPU) {
    sqccb::initialize_naive_cpu_backend(config);

    return 0;
  } else if (config->backend == SXT_NAIVE_BACKEND_GPU) {
    sqccb::initialize_naive_gpu_backend(config);

    return 0;
  } else if (config->backend == SXT_PIPPENGER_BACKEND_CPU) {
    sqccb::initialize_pippenger_cpu_backend(config);

    return 0;
  }

  return 1;
}
