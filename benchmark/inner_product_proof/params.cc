#include "params.h"

#include <cstdlib>
#include <sstream>
#include <string_view>

#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"
#include "sxt/cbindings/backend/cpu_backend.h"
#include "sxt/cbindings/backend/gpu_backend.h"

namespace sxt::bncip {
//--------------------------------------------------------------------------------------------------
// read_params
//--------------------------------------------------------------------------------------------------
void read_params(params& params, int argc, char* argv[]) noexcept {
  if (argc < 4) {
    std::cerr << "Usage: benchmark "
              << "<cpu|gpu> "
              << "<n> "
              << "<num_samples>\n";
    std::exit(-1);
  }
  std::string_view backend{argv[1]};
  params.backend_name = backend;
  if (backend == "cpu") {
    params.backend = std::make_unique<cbnbck::cpu_backend>();
  } else if (backend == "gpu") {
    params.backend = std::make_unique<cbnbck::gpu_backend>();
  } else {
    std::ostringstream oss;
    oss << "unknown backend: " << backend << "\n";
    baser::panic(oss.str());
  }
  params.n = static_cast<size_t>(std::atoi(argv[2]));
  SXT_RELEASE_ASSERT(params.n > 0);
  params.iterations = static_cast<size_t>(std::atoi(argv[3]));
  SXT_RELEASE_ASSERT(params.iterations > 0);
}
} // namespace sxt::bncip
