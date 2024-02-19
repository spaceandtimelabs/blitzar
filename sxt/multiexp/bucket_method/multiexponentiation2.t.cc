#include "sxt/multiexp/bucket_method/multiexponentiation2.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/field51/type/literal.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk;
using c21t::operator""_c21;
using f51t::operator""_f51;

TEST_CASE("we can compute a multiexponentiation") {
  std::vector<bascrv::element97> res(1);
  std::vector<bascrv::element97> generators;
  std::vector<const uint8_t*> scalars;
  const unsigned element_num_bytes = 32;

  SECTION("we can compute a multiexponentiation with no elements") {
    res.clear();
    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    REQUIRE(fut.ready());
  }

  SECTION("we can compute a multiexponentiation with a single zero element") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    generators = {33u};
    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 0u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 1") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1;
    scalars = {scalars1.data()};
    generators = {12u};
    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 12u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 2") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 2;
    scalars = {scalars1.data()};
    generators = {12u};
    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 24u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 256") {
    std::vector<uint8_t> scalars1(32);
    scalars1[1] = 1;
    scalars = {scalars1.data()};
    generators = {12u};
    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 256u * 12u);
  }

  SECTION("we can compute a multiexponentiation with 2 generators") {
    std::vector<uint8_t> scalars1(64);
    scalars1[0] = 2;
    scalars1[32] = 3;
    scalars = {scalars1.data()};

    generators = {12u, 34u};

    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * 12u + 3u * 34u);
  }

  SECTION("we can compute a multiexponetiation with multiple outputs") {
    res.resize(2);

    std::vector<uint8_t> scalars1(64);
    scalars1[0] = 2;
    scalars1[32] = 3;

    std::vector<uint8_t> scalars2(64);
    scalars2[0] = 7;
    scalars2[32] = 4;

    scalars = {scalars1.data(), scalars2.data()};

    generators = {12u, 34u};

    auto fut = multiexponentiate2<bascrv::element97>(res, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * 12u + 3u * 34u);
    REQUIRE(res[1] == 7u * 12u + 4u * 34u);
  }
}
