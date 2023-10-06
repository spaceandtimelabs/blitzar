#include "sxt/multiexp/bucket_method/multiexponentiation.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/execution/schedule/scheduler.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can compute a multiexponentiation") {
  std::vector<bascrv::element97> res(1);
  std::vector<bascrv::element97> generators;
  std::vector<const uint8_t*> exponents;

  SECTION("we can compute a multiexponentiation with no elements") {
    res.clear();
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    REQUIRE(fut.ready());
  }

  SECTION("we can compute a multiexponentiation with a single zero element") {
    uint8_t scalar_data[32] = {};
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 0u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 1") {
    uint8_t scalar_data[32] = {};
    scalar_data[0] = 1;
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 12u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 2") {
    uint8_t scalar_data[32] = {};
    scalar_data[0] = 2;
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 24u);
  }

  SECTION("we can compute a multiexponentiation with 2 generators") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 2;
    scalar_data[32] = 3;
    exponents.push_back(scalar_data);

    generators = {12u, 34u};

    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * 12u + 3u * 34u);
  }
}
