#include "sxt/seqcommit/cbindings/get_one_commit.h"

#include <iostream>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/cbindings/backend.h"
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

//--------------------------------------------------------------------------------------------------
// sxt_get_one_commit
//--------------------------------------------------------------------------------------------------
int sxt_get_one_commit(struct sxt_ristretto* one_commit, uint64_t n) {
  if (!sxt::sqccb::is_backend_initialized()) {
    std::cerr << "ABORT: backend uninitialized in the `sxt_get_one_commit` c binding function"
              << std::endl;
    std::abort();
  }

  if (one_commit == nullptr) {
    std::cerr << "ABORT: one_commit input to `sxt_get_one_commit` c binding function is null"
              << std::endl;
    std::abort();
  }

  reinterpret_cast<sxt::c21t::element_p3*>(one_commit)[0] =
      sxt::sqcgn::get_precomputed_one_commit(n);

  return 0;
}
