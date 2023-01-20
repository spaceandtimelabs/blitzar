#include "cbindings/get_one_commit.h"

#include <iostream>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_get_one_commit
//--------------------------------------------------------------------------------------------------
int sxt_get_one_commit(struct sxt_ristretto* one_commit, uint64_t n) {
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized(),
                     "backend uninitialized in the `sxt_get_one_commit` c binding function");
  SXT_RELEASE_ASSERT(one_commit != nullptr,
                     "one_commit input to `sxt_get_one_commit` c binding function is null");

  reinterpret_cast<sxt::c21t::element_p3*>(one_commit)[0] =
      sxt::sqcgn::get_precomputed_one_commit(n);

  return 0;
}
