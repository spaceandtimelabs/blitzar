#include "sxt/base/num/fast_random_number_generator.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt::basn;

TEST_CASE(
    "fast_random_number_generator can be used to quickly generate a sequence "
    "of random numbers") {
  fast_random_number_generator generator1{1, 2};
  auto x1 = generator1();
  auto x2 = generator1();
  REQUIRE(x1 != x2);
}
