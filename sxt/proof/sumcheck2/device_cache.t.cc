#include "sxt/proof/sumcheck2/device_cache.h"

#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;


TEST_CASE("we can cache device values that don't change as a proof is computed") {
  using T = s25t::element;
  std::vector<std::pair<s25t::element, unsigned>> product_table;
  std::vector<unsigned> product_terms;

  basdv::stream stream;

  basct::cspan<std::pair<s25t::element, unsigned>> product_table_dev;
  basct::cspan<unsigned> product_terms_dev;

  SECTION("we can access values from device memory") {
    product_table = {{0x123_s25, 0}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    cache.lookup(product_table_dev, product_terms_dev, stream);

    std::vector<std::pair<s25t::element, unsigned>> product_table_p(product_table.size());
    basdv::async_copy_device_to_host(product_table_p, product_table_dev, stream);

    std::vector<unsigned> product_terms_p(product_terms.size());
    basdv::async_copy_device_to_host(product_terms_p, product_terms_dev, stream);

    basdv::synchronize_stream(stream);
    REQUIRE(product_table_p == product_table);
    REQUIRE(product_terms_p == product_terms);
  }

  SECTION("we can clear the device cache") {
    product_table = {{0x123_s25, 0}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    cache.lookup(product_table_dev, product_terms_dev, stream);

    std::vector<std::pair<s25t::element, unsigned>> product_table_p(product_table.size());
    basdv::async_copy_device_to_host(product_table_p, product_table_dev, stream);

    std::vector<unsigned> product_terms_p(product_terms.size());
    basdv::async_copy_device_to_host(product_terms_p, product_terms_dev, stream);

    auto data = cache.clear();
    basdv::async_copy_device_to_host(product_table_p, data->product_table, stream);
    basdv::async_copy_device_to_host(product_terms_p, data->product_terms, stream);
    basdv::synchronize_stream(stream);
    REQUIRE(product_table_p == product_table);
    REQUIRE(product_terms_p == product_terms);
  }
}
