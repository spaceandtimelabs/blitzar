#include "cbindings/backend.h"

#include <iostream>

#include "sxt/base/device/property.h"
#include "sxt/base/error/assert.h"
#include "sxt/cbindings/backend/computational_backend.h"
#include "sxt/cbindings/backend/cpu_backend.h"
#include "sxt/cbindings/backend/gpu_backend.h"
#include "sxt/seqcommit/generator/precomputed_initializer.h"

using namespace sxt;

namespace sxt::cbn {
//--------------------------------------------------------------------------------------------------
// backend
//--------------------------------------------------------------------------------------------------
static cbnbck::computational_backend* backend = nullptr;

//--------------------------------------------------------------------------------------------------
// initialize_cpu_backend
//--------------------------------------------------------------------------------------------------
static void initialize_cpu_backend(const sxt_config* config) noexcept {
  backend = cbnbck::get_cpu_backend();
  sqcgn::init_precomputed_components(config->num_precomputed_generators, false);
}

//--------------------------------------------------------------------------------------------------
// initialize_gpu_backend
//--------------------------------------------------------------------------------------------------
static void initialize_gpu_backend(const sxt_config* config) noexcept {
  int num_devices = basdv::get_num_devices();

  if (num_devices == 0) {
    initialize_cpu_backend(config);

    // this message is used only to warn the user that gpu will not be used
    std::cout << "WARN: Using pippenger cpu instead of naive gpu backend." << std::endl;

    return;
  }

  backend = cbnbck::get_gpu_backend();
  sqcgn::init_precomputed_components(config->num_precomputed_generators, true);
}

//--------------------------------------------------------------------------------------------------
// is_backend_initialized
//--------------------------------------------------------------------------------------------------
bool is_backend_initialized() noexcept { return backend != nullptr; }

//--------------------------------------------------------------------------------------------------
// get_backend
//--------------------------------------------------------------------------------------------------
cbnbck::computational_backend* get_backend() noexcept {
  SXT_RELEASE_ASSERT(backend != nullptr);

  return backend;
}

//--------------------------------------------------------------------------------------------------
// reset_backend_for_testing
//--------------------------------------------------------------------------------------------------
void reset_backend_for_testing() noexcept {
  SXT_RELEASE_ASSERT(backend != nullptr);

  backend = nullptr;
}
} // namespace sxt::cbn

//--------------------------------------------------------------------------------------------------
// sxt_init
//--------------------------------------------------------------------------------------------------
int sxt_init(const sxt_config* config) {
  SXT_RELEASE_ASSERT(config != nullptr, "config input to `sxt_init` c binding function is null");
  SXT_RELEASE_ASSERT(cbn::backend == nullptr,
                     "trying to reinitialize the backend in the `sxt_init` c binding function");

  if (config->backend == SXT_GPU_BACKEND) {
    cbn::initialize_gpu_backend(config);

    return 0;
  } else if (config->backend == SXT_CPU_BACKEND) {
    cbn::initialize_cpu_backend(config);

    return 0;
  }

  return 1;
}
