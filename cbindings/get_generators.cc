#include "cbindings/get_generators.h"

#include <iostream>

#include "cbindings/backend.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_get_generators
//--------------------------------------------------------------------------------------------------
int sxt_get_generators(struct sxt_ristretto* generators, uint64_t num_generators,
                       uint64_t offset_generators) {
  if (!sxt::cbn::is_backend_initialized()) {
    std::cerr << "ABORT: backend uninitialized in the `sxt_get_generators` c binding function"
              << std::endl;
    std::abort();
  }

  // we ignore the function call when zero generators are specified.
  // in this case, generators can be null
  if (num_generators == 0) {
    return 0;
  }

  // generators must not be null since we need
  // to write the results back into the generators pointer
  if (num_generators > 0 && generators == nullptr) {
    return 1;
  }

  basct::span<c21t::element_p3> generators_result(reinterpret_cast<c21t::element_p3*>(generators),
                                                  num_generators);

  auto backend = sxt::cbn::get_backend();
  backend->get_generators(generators_result, offset_generators);

  return 0;
}
