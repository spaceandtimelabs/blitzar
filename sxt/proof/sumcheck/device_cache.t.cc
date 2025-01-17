#include "sxt/proof/sumcheck/device_cache.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/literal.h"
using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can cache device values that don't change as a proof is computed") {
  std::vector<std::pair<s25t::element, unsigned>> product_table;
  std::vector<unsigned> product_terms;

  basdv::stream stream;

  basct::cspan<std::pair<s25t::element, unsigned>> product_table_dev;
  basct::cspan<unsigned> product_terms_dev;

  SECTION("we can access values from device memory") { 
    product_table = {{0x123_s25, 0}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    cache.lookup(product_table_dev, product_terms_dev, stream);
    /* void lookup(basct::cspan<std::pair<s25t::element, unsigned>>& product_table, */
    /*             basct::cspan<unsigned>& product_terms, basdv::stream& stream) noexcept; */
  }
  /* device_cache(basct::cspan<std::pair<s25t::element, unsigned>> product_table, */
  /*              basct::cspan<unsigned> product_terms) noexcept; */
}
