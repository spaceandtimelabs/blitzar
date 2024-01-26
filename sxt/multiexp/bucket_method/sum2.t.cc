#include "sxt/multiexp/bucket_method/sum2.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxbk;

using E = bascrv::element97;

TEST_CASE("we can compute bucket sums") {
  memmg::managed_array<E> sums;

  basdv::stream stream;
  std::vector<E> generators;
  std::vector<const uint8_t*> scalars(1);

  unsigned element_num_bytes = 1u;
  unsigned bit_width = 2u;

  SECTION("we handle the empty case") {
    compute_bucket_sums<E>(sums, stream, generators, scalars, element_num_bytes, bit_width);
/* template <bascrv::element T> */
/* void compute_bucket_sums(basct::span<T> sums, const basdv::stream& stream, */
/*                          basct::cspan<T> generators, basct::cspan<const uint8_t*> scalars, */
/*                          unsigned element_num_bytes, unsigned bit_width) noexcept { */
/* template <bascrv::element T> */
/* xena::future<> compute_bucket_sums2(basct::span<T> sums, basct::cspan<T> generators, */
/*                                     basct::cspan<const uint8_t*> scalars, */
/*                                     unsigned element_num_bytes, unsigned bit_width) noexcept { */
  }
}
