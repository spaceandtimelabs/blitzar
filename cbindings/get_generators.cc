#include "cbindings/get_generators.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_get_generators
//--------------------------------------------------------------------------------------------------
int sxt_get_generators(struct sxt_ristretto* generators, uint64_t num_generators,
                       uint64_t offset_generators) {
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized(),
                     "backend uninitialized in the `sxt_get_generators` c binding function");

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

  auto backend = sxt::cbn::get_backend();

  std::vector<c21t::element_p3> temp_generators;
  auto precomputed_generators =
      backend->get_precomputed_generators(temp_generators, num_generators, offset_generators);
  std::copy_n(precomputed_generators.begin(), num_generators,
              reinterpret_cast<c21t::element_p3*>(generators));

  return 0;
}
