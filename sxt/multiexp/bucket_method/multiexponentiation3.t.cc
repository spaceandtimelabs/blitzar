#include "sxt/multiexp/bucket_method/multiexponentiation3.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;
using namespace sxt::mtxbk;

using E = bascrv::element97;
using c21t::operator""_c21;

TEST_CASE("we can compute multiexponentiations") {
  memmg::managed_array<E> res(1);
  multiexponentiate_options3 options{
    .min_chunk_size = 1u,
    .max_chunk_size = 100u,
    .bit_width = 4u,
    /* .bit_width = 2u, */
    .split_factor = 1u,
  };
  memmg::managed_array<const uint8_t*> scalars;
  unsigned element_num_bytes = 32;

  memmg::managed_array<E> generators;

#if 0
  SECTION("we handle the case of a single scalar of 1") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1u;
    scalars = {scalars1.data()};
    generators = {33u};
    auto fut = multiexponentiate3<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 33u);
  }

  SECTION("we handle the case of a single scalar of 2") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 2u;
    scalars = {scalars1.data()};
    generators = {33u};
    auto fut = multiexponentiate3<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 66u);
  }

  SECTION("we handle two elements") {
    std::vector<uint8_t> scalars1(64);
    scalars1[0] = 2u;
    scalars1[32] = 7u;
    scalars = {scalars1.data()};
    generators = {33u, 53u};
    auto fut = multiexponentiate3<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 33u * 2u + 53u * 7u);
  }
#endif

#if 0
  SECTION("we handle two scalars using two chunks") {
    options.split_factor = 2u;
    memmg::managed_array<uint8_t> scalars1 = {2u, 7u};
    scalars = {scalars1.data()};
    generators = {33u, 53u};
    auto fut = multiexponentiate<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 33u * 2u + 53u * 7u);
  }

  SECTION("we handle scalars of multiple bytes") {
    element_num_bytes = 2u;
    memmg::managed_array<uint8_t> scalars1 = {0u, 1u};
    scalars = {scalars1.data()};
    generators = {33u};
    auto fut = multiexponentiate<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == (1u << 8u) * 33u);
  }

  SECTION("we handle a scalar of negative 1") {
    memmg::managed_array<uint8_t> scalars1 = {96u};
    scalars = {scalars1.data()};
    generators = {33u};
    auto fut = multiexponentiate<E>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == (97u - 33u));
  }
#endif
}

#if 0
TEST_CASE("we can compute multiexponentiations with curve21 elements") {
  std::vector<c21t::element_p3> res(1);
  std::vector<c21t::element_p3> generators;
  std::vector<const uint8_t*> scalars;
  multiexponentiate_options3 options{
    .min_chunk_size = 1u,
    .max_chunk_size = 100u,
    .bit_width = 8u,
    /* .bit_width = 2u, */
    .split_factor = 1u,
  };
  unsigned element_num_bytes = 32;

  SECTION("we can compute a multiexponentiation with a single element of 1") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1u;
    scalars = {scalars1.data()};

    generators = {0x123_c21};
    auto fut =
        multiexponentiate3<c21t::element_p3>(res, options, generators, scalars, element_num_bytes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 0x123_c21);
  }
}
#endif
